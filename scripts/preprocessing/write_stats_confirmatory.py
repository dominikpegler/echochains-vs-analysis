#!/usr/bin/env python3
"""Incremental preprocessing: confirmatory chain logs -> metric/axis exports.

Walker semantics: there are no model/task/condition filters. The script scans
every cell under ``$DATA_DIR/data/vs_data/logs/<model>/confirmatory/<task>/<condition>/``
(``<model>`` may contain a slash, e.g. an HF id ``org/name``)
and processes each complete chain log that has not been processed yet.

Per chain (``MSG_<seed>_chain_<n>[_rep<k>].json``) the script computes the
pilot's cumulative metrics plus projections onto the five semantic axes and
writes one parquet shard under

    $DATA_DIR/data/vs_data/stats/<model>/<config>/<task>/<condition>/shards/

The config folder is derived from the decoding recorded in the log
(``model_default`` when temperature/top-p are absent, else
``temp<t>_topp<p>``), mirroring the pilot's per-config layout even though
each confirmatory model runs a single decoding setting.

A chain counts as processed only when its shard and an adjacent
``<shard>.done.json`` sidecar exist and match the current log fingerprint and
preprocessing version. New logs are processed on the next run, changed logs
are recomputed, and everything else is skipped, so the script is safe to run
repeatedly while collection is in progress.

After a cell gains shards (or its CSV exports are missing), the exports are
rebuilt from all valid shards in the cell:

    metrics_cumulative_long.csv
    metrics_last_hop_cumulative.csv
    axis_cumulative_long.csv      (hop 0 .. 200, five axes)
    coverage.json

Only cumulative metrics are produced; the pilot's "local" kind has no
consumer and is not computed here.

Requires CUDA: the pilot's DeBERTa-large NLI runs in fp16 (the shared loader
calls ``.half()`` unconditionally), so compute mode is not supported on CPU.
``--dry-run`` works without a GPU and without loading any model.

Usage:
    python scripts/preprocessing/write_stats_confirmatory.py --dry-run
    python scripts/preprocessing/write_stats_confirmatory.py
    python scripts/preprocessing/write_stats_confirmatory.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# get_data_dir() calls load_dotenv() from inside echochain.utils, where the
# caller-anchored search lands in the package tree instead of the repo root.
# Anchor the .env lookup at this script explicitly.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HOPS = 200  # transmitted hops per chain (design lock)
CHAINS_PER_SEED = 5
PREPROC_VERSION = "1"
METRICS_VERSION = "v1.0.0"

TASK_BAND = {
    "paraphrase": "short",
    "improve": "short",
    "take_inspiration": "short",
    "expand+compress": "long",
}

METRICS = [
    "delta_len",
    "ratio_len",
    "jaccard",
    "entity_f1",
    "triple_f1",
    "role_swap",
    "cosine",
    "entail_ab",
    "entail_ba",
    "entail_consistent",
    "k_unique",
]

NAME_RE = re.compile(
    r"^MSG_(?P<seed>.+)_chain_(?P<chain>\d+)(?:_rep(?P<rep>\d+))?\.json$"
)


# ---------------------------------------------------------------------------
# Paths and discovery
# ---------------------------------------------------------------------------


def get_paths():
    from echochain.utils import get_data_dir

    data_dir = Path(get_data_dir())
    return (
        data_dir / "data" / "vs_data" / "logs",
        data_dir / "data" / "vs_data" / "stats",
    )


def discover_cells(logs_root: Path):
    """Yield (model, task, condition, logs_dir) for every cell on disk.

    The model is the path between the logs root and ``confirmatory/``, so HF
    ids such as ``hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`` map to
    a nested directory (and to the same nested stats directory).
    """
    if not logs_root.is_dir():
        return
    for conf_dir in sorted(logs_root.glob("**/confirmatory")):
        if not conf_dir.is_dir():
            continue
        model = conf_dir.parent.relative_to(logs_root).as_posix()
        for task_dir in sorted(p for p in conf_dir.iterdir() if p.is_dir()):
            for cond_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                yield model, task_dir.name, cond_dir.name, cond_dir


def discover_logs(logs_dir: Path):
    """Sorted chain logs in a cell, excluding FAILED records and partials."""
    return sorted(p for p in logs_dir.glob("MSG_*_chain_*.json"))


def parse_name(path: Path):
    m = NAME_RE.match(path.name)
    if not m:
        return None
    return {
        "seed": m.group("seed"),
        "chain": int(m.group("chain")),
        "rep": int(m.group("rep")) if m.group("rep") is not None else -1,
    }


# ---------------------------------------------------------------------------
# Chain validation
# ---------------------------------------------------------------------------


def validate_chain(chain, model, task, condition, seed):
    """Return None when the log is a complete, consistent chain, else a reason."""
    if not isinstance(chain, list) or not chain:
        return "empty or non-list log"
    expected = HOPS + 1
    if len(chain) != expected:
        return f"incomplete: {len(chain)} hops (expected {expected})"
    for i, hop in enumerate(chain):
        if not isinstance(hop, dict):
            return f"hop {i} is not an object"
        if hop.get("hop") != i:
            return f"hop numbering broken at index {i}"
        if not isinstance(hop.get("text"), str) or not hop["text"].strip():
            return f"missing text at hop {i}"
    for key, want in (
        ("model", model),
        ("task", task),
        ("condition", condition),
        ("seed_id", seed),
    ):
        seen = {h.get(key) for h in chain if h.get(key) is not None}
        if seen and seen != {want}:
            return f"metadata mismatch: {key}={sorted(map(str, seen))} != {want!r}"
    return None


def derive_config(chain):
    hop0 = chain[0]
    temp, topp = hop0.get("temperature"), hop0.get("top_p")
    if temp is None or topp is None:
        return "model_default"
    return f"temp{float(temp):g}_topp{float(topp):g}"


# ---------------------------------------------------------------------------
# Shard bookkeeping
# ---------------------------------------------------------------------------


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_paths(cell_dir: Path, path: Path):
    stem = path.stem
    shards = cell_dir / "shards"
    return shards / f"{stem}.parquet", shards / f"{stem}.done.json"


def is_current(shard: Path, sidecar: Path, path: Path):
    """Return (current, reason).

    Fast path: size and mtime match the sidecar, no file read. If either
    changed, compare the content hash; an unchanged hash (touch/rewrite with
    identical bytes) still counts as current.
    """
    if not shard.exists() or not sidecar.exists():
        return False, "missing shard or sidecar"
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "unreadable sidecar"
    if meta.get("version") != PREPROC_VERSION:
        return False, f"preprocessing version {meta.get('version')} != {PREPROC_VERSION}"
    st = path.stat()
    if (
        meta.get("source_size") == st.st_size
        and meta.get("source_mtime_ns") == st.st_mtime_ns
    ):
        return True, ""
    if meta.get("source_sha256") == fingerprint(path):
        return True, ""
    return False, "source log changed"


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def load_models():
    """Import and instantiate the heavy models; requires CUDA."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for compute mode: the pilot's DeBERTa-large NLI "
            "runs in fp16 and the shared loader calls .half() unconditionally. "
            "Run this script on the GPU workstation, or use --dry-run here."
        )

    from echochain import metrics_utils as M
    from echochain.semantic_axes import AXES, EMBEDDING_MODEL, project_onto_axis

    # Fail early and clearly if the spacy/NLI models are unavailable.
    M.get_spacy()
    M.get_nli()

    return {
        "embed": EMBEDDING_MODEL,
        "axes": AXES,
        "project": project_onto_axis,
        "token_delta": M.token_delta,
        "token_expansion_ratio": M.token_expansion_ratio,
        "jaccard": M.jaccard,
        "entity_retention": M.entity_retention,
        "triple_retention": M.triple_retention,
        "role_swap_rate": M.role_swap_rate,
        "cosine_similarity": M.cosine_similarity,
        "entailment_consistency": M.entailment_consistency,
    }


def compute_rows(chain, model, config, task, condition, seed, chain_idx, rep, h):
    """One long DataFrame of cumulative metrics + axis scores for one chain.

    Metric rows cover hops 1..200 (pilot layout; hop 0 is the seed). Axis rows
    cover hops 0..200 so drift slopes can include the seed position.
    """
    texts = [hop["text"] for hop in chain]
    emb = h["embed"].encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )

    meta = dict(
        model=model,
        config=config,
        task=task,
        condition=condition,
        msg_id=seed,
        chain=chain_idx,
        rep=int(rep),
    )
    rows = []

    for i in range(len(texts)):
        for axis, (start, end) in h["axes"].items():
            rows.append(
                dict(
                    **meta,
                    hop=i,
                    kind="cumulative",
                    metric=f"axis:{axis}",
                    value=float(h["project"](emb[i], start, end)),
                    ver=METRICS_VERSION,
                )
            )

    seed_text, seed_vec = texts[0], emb[0]
    seen = {seed_text}
    for i in range(1, len(texts)):
        txt = texts[i]
        er = h["entity_retention"](seed_text, txt)
        tr = h["triple_retention"](seed_text, txt)
        ec = h["entailment_consistency"](seed_text, txt, thresh=0.5)
        seen.add(txt)
        values = {
            "delta_len": h["token_delta"](seed_text, txt),
            "ratio_len": h["token_expansion_ratio"](seed_text, txt),
            "jaccard": h["jaccard"](seed_text, txt),
            "entity_f1": er["f1"],
            "triple_f1": tr["f1"],
            "role_swap": h["role_swap_rate"](seed_text, txt),
            "cosine": h["cosine_similarity"](seed_vec, emb[i]),
            "entail_ab": ec["entail_a_to_b"],
            "entail_ba": ec["entail_b_to_a"],
            "entail_consistent": ec["consistent"],
            "k_unique": len(seen),
        }
        for metric, value in values.items():
            rows.append(
                dict(
                    **meta,
                    hop=i,
                    kind="cumulative",
                    metric=metric,
                    value=float(value),
                    ver=METRICS_VERSION,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def export_cell(cell_dir: Path):
    """Rebuild CSV exports from all shards in a cell."""
    shard_files = sorted((cell_dir / "shards").glob("*.parquet"))
    if not shard_files:
        return
    df = pd.concat([pd.read_parquet(p) for p in shard_files], ignore_index=True)
    if "rep" not in df.columns:
        df["rep"] = -1
    df["rep"] = df["rep"].fillna(-1).astype(int)

    axis_mask = df["metric"].str.startswith("axis:")
    metrics = df[~axis_mask].copy()
    axis = df[axis_mask].copy()

    _write_csv(metrics, cell_dir / "metrics_cumulative_long.csv")

    keys = ["model", "config", "task", "condition", "msg_id", "chain", "rep", "metric"]
    idx = metrics.groupby(keys, dropna=False)["hop"].idxmax()
    _write_csv(metrics.loc[idx], cell_dir / "metrics_last_hop_cumulative.csv")

    if not axis.empty:
        axis["axis"] = axis["metric"].str.slice(len("axis:"))
        wide = axis.pivot_table(
            index=[
                "model",
                "config",
                "task",
                "condition",
                "msg_id",
                "chain",
                "rep",
                "hop",
                "kind",
                "ver",
            ],
            columns="axis",
            values="value",
            dropna=False,
        ).reset_index()
        wide.columns.name = None
        head = ["model", "config", "task", "condition", "msg_id", "chain", "rep", "hop", "kind", "ver"]
        axis_cols = [c for c in wide.columns if c not in head]
        _write_csv(wide[head + sorted(axis_cols)], cell_dir / "axis_cumulative_long.csv")


def write_coverage(cell_dir: Path, entries, model, task, condition, seeds_by_band):
    """Count what exists on disk for the cell; used by the analysis notebook.

    ``entries`` come from the discovery pass and already carry the validation
    verdict, so no log is re-read here.
    """
    incomplete = [
        f"{e['path'].name}: {e['problem']}" for e in entries if e["problem"] is not None
    ]
    complete = sum(1 for e in entries if e["problem"] is None)
    logs_dir = entries[0]["path"].parent if entries else None
    failed = len(list(logs_dir.glob("FAILED_MSG_*.json"))) if logs_dir else 0
    partial = len(list(logs_dir.glob("MSG_*_chain_*.json.partial"))) if logs_dir else 0
    band = TASK_BAND.get(task)
    expected_seeds = len(seeds_by_band.get(band, [])) if seeds_by_band else None
    shards = sorted((cell_dir / "shards").glob("*.parquet")) if (cell_dir / "shards").is_dir() else []

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preprocessing_version": PREPROC_VERSION,
        "metrics_version": METRICS_VERSION,
        "model": model,
        "task": task,
        "condition": condition,
        "expected": {
            "seeds": expected_seeds,
            "chains_per_seed": CHAINS_PER_SEED,
            "chains": expected_seeds * CHAINS_PER_SEED if expected_seeds else None,
            "hops_per_chain": HOPS,
        },
        "found": {
            "log_files": len(entries),
            "complete_logs": complete,
            "incomplete": incomplete,
            "failed_records": failed,
            "partial_logs": partial,
        },
        "shards": {"total": len(shards)},
    }
    atomic_write_text(
        cell_dir / "coverage.json", json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Seeds (for coverage denominators)
# ---------------------------------------------------------------------------


def default_seeds_file() -> Path:
    from echochain.utils import get_data_dir

    return (
        Path(get_data_dir())
        / "data"
        / "corpus"
        / "sampler"
        / "vs_confirmatory_seeds_final.jsonl"
    )


def load_seeds(path: Path):
    if not path.exists():
        print(f"warning: seeds file not found ({path}); expected counts omitted")
        return {}
    by_band: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_band.setdefault(rec.get("band", "?"), []).append(rec["seed_id"])
    return by_band


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_plan(logs_root: Path, stats_root: Path):
    """One discovery pass: every chain log with its parsed identity and state.

    Returns a dict keyed by (model, config, task, condition) with the entries
    needed by the processing loop and the coverage writer.
    """
    plan: dict[tuple, dict] = {}
    for model, task, condition, logs_dir in discover_cells(logs_root):
        for path in discover_logs(logs_dir):
            parsed = parse_name(path)
            if parsed is None:
                print(f"warning: unparsable chain log name: {path}")
                continue
            try:
                chain = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                config = "unreadable"
                problem = f"unreadable ({exc.__class__.__name__})"
            else:
                config = derive_config(chain) if chain else "unreadable"
                problem = validate_chain(chain, model, task, condition, parsed["seed"])
            cell_dir = stats_root / model / config / task / condition
            shard, sidecar = shard_paths(cell_dir, path)
            current, reason = is_current(shard, sidecar, path)
            key = (model, config, task, condition)
            cell = plan.setdefault(
                key,
                {
                    "model": model,
                    "config": config,
                    "task": task,
                    "condition": condition,
                    "logs_dir": logs_dir,
                    "cell_dir": cell_dir,
                    "entries": [],
                },
            )
            cell["entries"].append(
                {
                    "path": path,
                    "parsed": parsed,
                    "problem": problem,
                    "current": current,
                    "reason": reason,
                }
            )
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute every chain")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list pending chains without loading models or writing anything",
    )
    ap.add_argument("--seeds-file", type=Path, default=None)
    args = ap.parse_args()

    logs_root, stats_root = get_paths()
    seeds_by_band = load_seeds(args.seeds_file or default_seeds_file())

    t0 = datetime.now(UTC)
    plan = build_plan(logs_root, stats_root)
    if not plan:
        print(f"no confirmatory chain logs under {logs_root}")
        return 0

    total_pending = 0
    for cell in plan.values():
        pending = [
            e
            for e in cell["entries"]
            if e["problem"] is None and (args.force or not e["current"])
        ]
        total_pending += len(pending)
        print(
            f"{cell['model']} / {cell['config']} / {cell['task']} / "
            f"{cell['condition']}: {len(cell['entries'])} logs, "
            f"{len(pending)} to process"
        )

    if args.dry_run:
        print(f"\ndry run: {total_pending} chain(s) would be processed")
        return 0

    if total_pending == 0:
        print("\nnothing to do")
        return 0

    print("\nloading models (CUDA required) ...")
    handles = load_models()
    print(f"models loaded in {(datetime.now(UTC) - t0).total_seconds():.1f}s")

    processed = 0
    for key in sorted(plan):
        cell = plan[key]
        cell_dir = cell["cell_dir"]
        touched = False
        for entry in cell["entries"]:
            path = entry["path"]
            if entry["problem"] is not None:
                print(f"  skip {path.name}: {entry['problem']}")
                continue
            if entry["current"] and not args.force:
                continue
            if not entry["current"] and entry["reason"] not in (
                "missing shard or sidecar",
                "",
            ):
                print(f"  {path.name}: recomputing ({entry['reason']})")

            raw = path.read_bytes()
            chain = json.loads(raw)
            parsed = entry["parsed"]
            shard, sidecar = shard_paths(cell_dir, path)
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "shards").mkdir(parents=True, exist_ok=True)
            df = compute_rows(
                chain,
                cell["model"],
                cell["config"],
                cell["task"],
                cell["condition"],
                parsed["seed"],
                parsed["chain"],
                parsed["rep"],
                handles,
            )
            stat = path.stat()
            atomic_write_parquet(df, shard)
            atomic_write_text(
                sidecar,
                json.dumps(
                    {
                        "version": PREPROC_VERSION,
                        "metrics_version": METRICS_VERSION,
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_size": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "hops": len(chain),
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                ),
            )
            touched = True
            processed += 1
            print(f"  wrote {shard.relative_to(stats_root)}")

        cell_dir.mkdir(parents=True, exist_ok=True)
        if touched or not (cell_dir / "metrics_cumulative_long.csv").exists():
            export_cell(cell_dir)
        write_coverage(
            cell_dir,
            cell["entries"],
            cell["model"],
            cell["task"],
            cell["condition"],
            seeds_by_band,
        )

    elapsed = (datetime.now(UTC) - t0).total_seconds()
    print(f"\nprocessed {processed} chain(s) in {elapsed / 60:.1f} min")
    print(f"stats root: {stats_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
