"""QC pass over confirmatory chain logs (Methods quality control).

READS written chain logs (no API access) and reports, per task x condition
cell:
1. Coverage: hops present per seed x chain; incomplete chains; crash/pause
   markers found in the cell directory.
2. Technical-failure rate: share of hops needing at least one retry
   (entry["attempts"] > 1); the Methods' pause-and-inspect flag at > 10%.
   Technical retries are counted separately from substantive performance.
3. VS collapse check (joint-list conditions): a hop is "peaked" when the cell
   has fewer than 2 candidates or the top normalized probability exceeds
   --collapse-p (default 0.95); a seed is flagged when more than
   --collapse-threshold (default 50%) of its hops are peaked.
4. NLI floor on hop 1: seed-to-hop1 entailment (DeBERTa-large-MNLI, the
   pilot's measure in echochain.metrics_utils) below --nli-floor (default
   0.3) flags the affected (seed, chain) pair for a rerun with a fresh
   replicate (--replicate 1 with the same command writes ..._rep1.json and
   never clobbers the original). The rerun list is written to the report;
   which pairs to rerun is a freeze-time decision, reported but not applied
   here. --skip-nli omits the model pass (torch-free inspection of the rest).

Usage:
  python py/qc_chains_confirmatory.py --model gpt-5.6-luna \
      --task paraphrase --condition vs_weighted [--skip-nli]
  python py/qc_chains_confirmatory.py --all-cells --skip-nli --model ...

Exit codes: 0 = no flags, 1 = flags found (or cells missing), 2 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

TASK_ORDER = ("paraphrase", "improve", "take_inspiration", "expand+compress")
CONDITION_ORDER = ("direct", "uniform_list", "vs_weighted")


# ---------------------------------------------------------------------------
# Cell inspection
# ---------------------------------------------------------------------------

def load_chain(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_cell(cell_dir: Path, task: str, condition: str, *, hops: int,
                 chains_expected: int, n_seeds: int) -> dict:
    """Compute coverage, technical-failure, and collapse statistics."""
    logs = sorted(cell_dir.glob("MSG_*_chain_*.json"))
    coverage = {"chains_found": len(logs),
                "chains_expected": n_seeds * chains_expected}
    incomplete = []
    retry_hops = 0
    total_hops = 0
    per_seed_peaked: dict[int, list[int]] = {}
    nli_hops: list[dict] = []

    def hop_peaked(entry: dict, p_star: float) -> bool:
        """True when any stage of the hop is degenerate (fewer than 2
        candidates) or its normalized probability mass concentrates above
        p_star (collapse check; probability applies to VS only)."""
        stages = entry.get("stages") or [entry]
        for stage_view in stages:
            candidates = stage_view.get("candidates") or []
            probabilities = stage_view.get("probabilities_normalized")
            if len(candidates) < 2:
                return True
            if probabilities and max(probabilities) > p_star:
                return True
        return False

    for path in logs:
        chain = load_chain(path)
        hop_entries = [e for e in chain if e.get("hop", 0) > 0]
        last_hop = max((e["hop"] for e in hop_entries), default=0)
        if last_hop < hops:
            incomplete.append({"file": path.name, "last_hop": last_hop})
        total_hops += len(hop_entries)
        retry_hops += sum(1 for e in hop_entries if e.get("attempts", 1) > 1)
        seed_id = chain[0].get("seed_id", 0) if chain else 0
        if condition == "vs_weighted":
            counts = (per_seed_peaked.setdefault(seed_id, [0, 0]))
            counts[0] += sum(1 for e in hop_entries if hop_peaked(e, 0.95))
            counts[1] += len(hop_entries)
        hop1 = next((e for e in chain if e.get("hop") == 1), None)
        if hop1 is not None and chain:
            nli_hops.append({"file": path.name,
                             "seed_id": chain[0].get("seed_id"),
                             "chain": hop1.get("chain"),
                             "text": hop1.get("text"),
                             "seed_text": chain[0].get("text")})
    technical_rate = (retry_hops / total_hops) if total_hops else 0.0
    report = {
        "task": task,
        "condition": condition,
        "cell_dir": str(cell_dir),
        "coverage": coverage,
        "incomplete_chains": incomplete,
        "pause_markers": sorted(p.name for p in cell_dir.glob(".paused_*")),
        "technical": {"hops_total": total_hops, "hops_retried": retry_hops,
                      "retry_rate": technical_rate,
                      "flag_over_10pct": technical_rate > 0.10},
    }
    if condition == "vs_weighted":
        shares = {seed_id: ((p / n) if n else 0.0)
                  for seed_id, (p, n) in per_seed_peaked.items()}
        report["collapse"] = {
            "share_by_seed": shares,
            "flagged_seeds": [sid for sid, share in shares.items()
                              if share > 0.5],
        }
    report["hop1_samples_for_nli"] = nli_hops
    return report


def run_nli_floor(report: dict, floor: float) -> dict:
    """Score hop-1 NLI (seed -> hop) and add the floor flags to the report.

    Uses echochain.metrics_utils.entailment_consistency (DeBERTa-large-MNLI),
    the pilot's measure: entail_a_to_b = the seed entails the hop.
    """
    from echochain.metrics_utils import entailment_consistency
    flagged = []
    for sample in report["hop1_samples_for_nli"]:
        ec = entailment_consistency(sample["seed_text"], sample["text"])
        ab = ec["entail_a_to_b"]
        if ab < floor:
            flagged.append({"file": sample["file"], "seed_id": sample["seed_id"],
                            "chain": sample["chain"],
                            "entail_ab": round(ab, 4)})
    report["nli_floor"] = {
        "floor": floor,
        "scored_hops": len(report["hop1_samples_for_nli"]),
        "flagged_rerun_pairs": flagged,
    }
    return report


def summarize(report: dict) -> str:
    lines = [f"=== {report['task']} / {report['condition']} ==="]
    cov = report["coverage"]
    lines.append(f"chains: {cov['chains_found']}/{cov['chains_expected']}; "
                 f"incomplete: {len(report['incomplete_chains'])}")
    tech = report["technical"]
    lines.append(f"retry rate: {tech['hops_retried']}/{tech['hops_total']} "
                 f"= {tech['retry_rate']:.1%}"
                 + (" FLAG (>10%)" if tech["flag_over_10pct"] else ""))
    if "collapse" in report:
        n_flag = len(report["collapse"]["flagged_seeds"])
        lines.append(f"collapse: {n_flag} seed(s) flagged "
                     f"(>50% peaked hops at p>0.95)")
    if "nli_floor" in report:
        nli = report["nli_floor"]
        lines.append(f"NLI floor {nli['floor']}: {len(nli['flagged_rerun_pairs'])} "
                     f"of {nli['scored_hops']} chain(s) flagged for a rerun")
    return "\n".join(lines)


def qc_for_cell(model: str, task: str, condition: str, logs_root: Path,
                hops: int, chains: int, n_seeds: int, *, skip_nli: bool,
                nli_floor: float, out_dir: Path | None) -> tuple[bool, dict]:
    cell_dir = logs_root / model / "confirmatory" / task / condition
    if not cell_dir.exists():
        print(f"MISSING cell directory: {cell_dir}")
        return True, {}
    report = inspect_cell(cell_dir, task, condition, hops=hops,
                          chains_expected=chains, n_seeds=n_seeds)
    if not skip_nli:
        try:
            run_nli_floor(report, nli_floor)
        except ImportError:
            print("  (NLI model unavailable; report written without the floor)")
    flagged = bool(report["incomplete_chains"] or report["pause_markers"]
                   or report["technical"]["flag_over_10pct"]
                   or report.get("collapse", {}).get("flagged_seeds")
                   or (report.get("nli_floor", {}).get("flagged_rerun_pairs")
                       if not skip_nli else []))
    payload = dict(report)
    if skip_nli:
        payload.pop("hop1_samples_for_nli", None)
    print(summarize(report))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"qc_{task}-{condition}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"report: {path}")
    return flagged, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--task", choices=TASK_ORDER)
    parser.add_argument("--condition", choices=CONDITION_ORDER)
    parser.add_argument("--logs-root")
    parser.add_argument("--hops", type=int, default=200)
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=90)
    parser.add_argument("--nli-floor", type=float, default=0.3)
    parser.add_argument("--collapse-threshold", type=float, default=0.5)
    parser.add_argument("--collapse-p", type=float, default=0.95)
    parser.add_argument("--skip-nli", action="store_true",
                        help="omit the DeBERTa NLI pass (torch unavailable)")
    parser.add_argument("--output-dir",
                        help="override the QC report directory "
                             "(default DATA_DIR/data/echodrift_confirmatory_qc)")
    args = parser.parse_args(argv)

    if args.all_cells:
        cells = [(t, c) for t in TASK_ORDER for c in CONDITION_ORDER]
    elif args.task and args.condition:
        cells = [(args.task, args.condition)]
    else:
        parser.error("give --task and --condition, or --all-cells")

    import os
    data_root = None
    if args.logs_root and args.output_dir:
        # explicit roots make DATA_DIR unnecessary (cluster runs)
        logs_root = Path(os.path.expanduser(args.logs_root))
        out_dir = Path(os.path.expanduser(args.output_dir))
    else:
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            raise SystemExit("DATA_DIR not set; give --logs-root and "
                             "--output-dir to run without it")
        data_root = Path(os.path.expanduser(data_dir))
        logs_root = (Path(os.path.expanduser(args.logs_root))
                     if args.logs_root
                     else data_root / "data" / "echodrift_logs")
        out_dir = (Path(os.path.expanduser(args.output_dir))
                   if args.output_dir
                   else data_root / "data" / "echodrift_confirmatory_qc"
                   / args.model)

    any_flag = False
    for task, condition in cells:
        flagged, _ = qc_for_cell(
            args.model, task, condition, logs_root, args.hops, args.chains,
            args.n_seeds, skip_nli=args.skip_nli, nli_floor=args.nli_floor,
            out_dir=out_dir)
        any_flag = any_flag or flagged
    return 1 if any_flag else 0


if __name__ == "__main__":
    sys.exit(main())