"""Confirmatory design-size and cost model for the VS paper (3-task lock).

Recomputes transmitted texts, prompt calls, and API token spend from the
locked design constants instead of hand arithmetic. The task set is the
2026-10-30 lock: paraphrase, take inspiration, expand+compress ("continue"
dropped, so no length cap applies).

Token counts split into MEASURED (exploratory pilot, vs_weighted logs,
2026-10-30 session) and ASSUMED. The ASSUMED entries are the dominant
uncertainty: expand+compress runs on the 100-200-word long-text band, which
the pilot never ran, so its response lengths are guesses. The planned small
long-text pilot replaces them; re-run this script after that pilot.

Prices: Luna list 0.20/1.20 USD per M tokens. The Azure global-batch
discount is ASSUMED at 50% and must be verified before the budget line is
quoted. Only the Luna runs cost API money; the open-weight panel models run
on the Vienna cluster.

Run:  python py/design_size_vs.py            (defaults = locked design)
      python py/design_size_vs.py --csv PATH (also write a CSV breakdown)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --- design constants (2026-10-30 lock) ------------------------------------
TASKS = ("paraphrase", "take_inspiration", "expand+compress")
CONDITIONS = ("direct", "multi_k_uniform", "vs_weighted", "vs_argmax")
# Prompt calls per hop stage under each condition. Multi-k uniform sends k
# separate task-prompt calls per hop (the paper's definition); the VS calls
# elicit k candidates in ONE call, so they also send one call per stage.
CALLS_PER_STAGE = {
    "direct": 1,
    "multi_k_uniform": 5,
    "vs_weighted": 1,
    "vs_argmax": 1,
}
SEEDS_PER_TASK = 50   # one shared sentence set serves paraphrase + take_inspiration
CHAINS = 9
HOPS = 200            # transmitted texts; the hop-0 seed is not transmitted
PANEL_MODELS = 4      # Luna + Llama-3-8B + Qwen-27B + Gemma-27B (names TBD)
VS_K = 5

# --- token assumptions -------------------------------------------------------
SEED_TOKENS = {"sentence": 30, "long": 200}   # ASSUMED midpoints of 10-40 and 100-200 words
VS_INSTR_BLOCK = 175   # MEASURED: VS input 205 minus a ~30-token seed (incl. system + JSON spec)
JSON_OVERHEAD = 75     # MEASURED calibration: 245 vs output = 5 x 34 + 75

@dataclass
class Stage:
    name: str
    instr_tokens: int
    out_tokens: int
    source: str  # "measured" or "assumed"

TASK_STAGES: dict[str, list[Stage]] = {
    "paraphrase": [
        Stage("paraphrase", 65, 34, "measured"),   # system+user block; 95 total - 30 seed
    ],
    "take_inspiration": [
        # Perez single turn, no system prompt; response length ASSUMED at the
        # paraphrase band (the task outputs a new text of similar length).
        Stage("take_inspiration", 25, 40, "assumed"),
    ],
    "expand+compress": [  # long-text band; lengths ASSUMED until the long-text pilot
        Stage("expand", 22, 250, "assumed"),
        Stage("compress", 22, 180, "assumed"),
    ],
}
TASK_BAND = {
    "paraphrase": "sentence",
    "take_inspiration": "sentence",
    "expand+compress": "long",
}

# --- prices -------------------------------------------------------------------
LUNA_IN_PRICE = 0.20    # USD per M tokens, list
LUNA_OUT_PRICE = 1.20   # USD per M tokens, list
BATCH_DISCOUNT = 0.50   # ASSUMED; verify the Azure global-batch discount
USD_PER_EUR = 0.92      # ASSUMED conversion

# --- model ----------------------------------------------------------------------
def stage_call_tokens(stage: Stage, band: str, condition: str) -> tuple[int, int]:
    """(input, output) tokens for ONE call of this stage under this condition."""
    seed = SEED_TOKENS[band]
    is_vs = condition.startswith("vs_")
    inp = (VS_INSTR_BLOCK if is_vs else stage.instr_tokens) + seed
    out = stage.out_tokens * VS_K + JSON_OVERHEAD if is_vs else stage.out_tokens
    return inp, out


def design_rows() -> list[dict]:
    rows = []
    for task in TASKS:
        band = TASK_BAND[task]
        stages = TASK_STAGES[task]
        per_hop_texts = CHAINS * HOPS  # transmitted texts per condition
        for condition in CONDITIONS:
            calls = per_hop_texts * CALLS_PER_STAGE[condition] * len(stages) * SEEDS_PER_TASK
            transmitted = per_hop_texts * SEEDS_PER_TASK
            in_tok = out_tok = 0
            details = []
            for stage in stages:
                cin, cout = stage_call_tokens(stage, band, condition)
                stage_calls = per_hop_texts * CALLS_PER_STAGE[condition] * SEEDS_PER_TASK
                in_tok += stage_calls * cin
                out_tok += stage_calls * cout
                details.append(
                    f"{stage.name}:{stage_calls:,}x({cin}+{cout})"
                    f"[{stage.source}]"
                )
            usd = (in_tok * LUNA_IN_PRICE + out_tok * LUNA_OUT_PRICE) / 1e6
            rows.append(
                dict(task=task, condition=condition, transmitted=transmitted,
                     calls=calls, in_tok=in_tok, out_tok=out_tok, usd=usd,
                     details=" ".join(details))
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default=None, help="write the breakdown CSV here")
    args = ap.parse_args()

    rows = design_rows()
    widths = (18, 16, 12, 12, 14, 14, 9)
    header = f"{'task':<{widths[0]}}{'condition':<{widths[1]}}{'transmitted':>{widths[2]}}" \
             f"{'calls':>{widths[3]}}{'in_tok(M)':>{widths[4]}}{'out_tok(M)':>{widths[5]}}{'USD':>{widths[6]}}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['task']:<{widths[0]}}{r['condition']:<{widths[1]}}"
              f"{r['transmitted']:>{widths[2]},}{r['calls']:>{widths[3]},}"
              f"{r['in_tok']/1e6:>{widths[4]}.1f}{r['out_tok']/1e6:>{widths[5]}.1f}"
              f"{r['usd']:>{widths[6]}.2f}")

    tr = sum(r["transmitted"] for r in rows)
    cl = sum(r["calls"] for r in rows)
    it = sum(r["in_tok"] for r in rows)
    ot = sum(r["out_tok"] for r in rows)
    usd = sum(r["usd"] for r in rows)
    print("-" * len(header))
    print(f"{'PER MODEL total':<{widths[0]+widths[1]}}{tr:>{widths[2]},}{cl:>{widths[3]},}"
          f"{it/1e6:>{widths[4]}.1f}{ot/1e6:>{widths[5]}.1f}{usd:>{widths[6]}.2f}")
    print(f"\nPanel: {PANEL_MODELS} models -> transmitted {tr*PANEL_MODELS:,}, "
          f"prompt calls {cl*PANEL_MODELS:,}. Only Luna is billed; the open-weight "
          f"models run on the Vienna cluster.")
    print(f"Luna list: {usd:,.0f} USD ~ {usd*USD_PER_EUR:,.0f} EUR; "
          f"with batch x{BATCH_DISCOUNT}: {usd*BATCH_DISCOUNT*USD_PER_EUR:,.0f} EUR "
          f"(discount ASSUMED, verify with Azure).")
    print(f"\nCost share expand+compress: "
          f"{sum(r['usd'] for r in rows if r['task']=='expand+compress')/usd:.0%} "
          f"(dominated by ASSUMED long-text VS response lengths).")

    # Locked-recount checks (handoff 2026-10-30): 1,080,000 transmitted and
    # ~2,880,000 prompt calls per model.
    assert tr == 1_080_000, tr
    assert cl == 2_880_000, cl
    print("\nrecount checks passed: transmitted 1,080,000; prompt calls 2,880,000 (per model)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()