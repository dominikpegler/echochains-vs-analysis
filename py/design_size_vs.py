"""Confirmatory design-size and cost model for the VS paper (3-condition lock).

Recomputes transmitted texts, prompt calls, and API token spend from the
design constants instead of hand arithmetic. The task set is the
2026-10-30 lock: paraphrase, improve, take inspiration, expand+compress
("continue" dropped, so no length cap applies). paraphrase, improve, and
take inspiration share the one 90-sentence seed set; expand+compress runs
on the long-text band. The conditions are the 2026-10-31 lock: Direct, the
joint-list uniform control (ONE call that generates k distinct candidates
WITHOUT probabilities; replaces the earlier five-separate-calls Multi-k
definition, whose transmitted distribution duplicates Direct), and VS
(probabilities, weight-sampled selection). Verbalized mode was dropped
from the confirmatory design for economy; no selection-rule hypothesis
(old H4) is preregistered.

DECODING (2026-10-31 decision, supersedes the locked 0.8/0.5): every model
runs at its provider-recommended decoding. gpt-6-luna (Azure) exposes NO
sampling parameters and runs at the vendor default; the open-weight arms
use each vendor model card's temperature/top-p recommendation (Llama-3.1-8B
0.6/0.9; Qwen3-32B non-thinking 0.7/0.8; Gemma-3-27b-it 1.0/0.95), with
top-k and repetition penalties excluded everywhere. Only the temperature +
top-p pair is set. Token estimates are unaffected (decode-dependent length
differences are small relative to the current ASSUMED margins).

DECODING RESOLUTION (2026-10-31, closed): the design keeps gpt-6-luna at the
vendor default; the mixed decoding within the panel is accepted and gets
declared in the prereg. gpt-5.6-luna is no longer referenced anywhere in the
design; the panel anchor is gpt-6-luna (snapshot gpt-6-luna-2026-09-22).

Token counts split into MEASURED (exploratory pilot, vs_weighted logs,
2026-10-30 session) and ASSUMED. The ASSUMED entries are the dominant
uncertainty: expand+compress runs on the 100-200-word long-text band, which
the pilot never ran. The planned small long-text pilot replaces them; the
2026-10-31 gpt-6-luna validation run measured the long-band JOINT outputs
~80% BELOW these assumptions (VS expand 728 vs assumed 1325 per-call output)
on fixture seeds, so treat the expand+compress USD column as a conservative
overestimate.

Prices: gpt-6-luna list prices $0.10 input / $0.50 output per M tokens
(published Standard tier VERIFIED 2026-10-31 against multiple trackers;
batch $0.05/$0.25, a flat 50% cut). These are the OpenAI rate-card prices;
parity on the Azure foundry-dompeg portal still needs the one-time check.
The Azure global-batch discount is ASSUMED at 50% and must be verified
before the budget line is quoted. Only the luna runs cost API money; the
open-weight panel models run on the Vienna cluster.

Run:  python py/design_size_vs.py            (defaults = locked design)
      python py/design_size_vs.py --csv PATH (also write a CSV breakdown)

Design basis: the user picked 90 seeds x 5 chains from the design comparison
(50x9 / 75x6 / 112x4 / 112x5 / 150x3). The power notebook is being rebuilt
against the final tests and the revised condition set, so treat the power
basis as provisional until the 2026-10-31 plan's step 4 is done.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --- design constants (2026-10-30/31 locks) ---------------------------------
TASKS = ("paraphrase", "improve", "take_inspiration", "expand+compress")
CONDITIONS = ("direct", "uniform_list", "vs_weighted")
# Prompt calls per hop stage: EVERY condition now sends ONE call per stage.
# The uniform_list control elicits k candidates jointly (no probabilities);
# the VS call elicits k candidates with probabilities. Selection happens
# after each call (uniform for the control; proportional to the normalized
# verbalized probabilities for VS).
CALLS_PER_STAGE = {
    "direct": 1,
    "uniform_list": 1,
    "vs_weighted": 1,
}
SEEDS_PER_TASK = 90   # 90x5 lock (2026-10-30): 30 seeds per genre; shared sentence
                      # set serves paraphrase + improve + take_inspiration
CHAINS = 5            # budget-neutral with the earlier 50x9 candidate (450 chains per
                      # condition per task), so totals and costs are unchanged
HOPS = 200            # transmitted texts; the hop-0 seed is not transmitted
PANEL_MODELS = 4      # gpt-6-luna (Azure) + Llama-3.1-8B + Qwen3-32B +
                      # Gemma-3-27b-it, 27B-class bf16 (names pinned 2026-10-31)
VS_K = 5

# --- token assumptions -------------------------------------------------------
SEED_TOKENS = {"sentence": 30, "long": 200}   # ASSUMED midpoints of 10-40 and 100-200 words
VS_INSTR_BLOCK = 175   # MEASURED: VS input 205 minus a ~30-token seed (incl. system + JSON spec)
JSON_OVERHEAD = 75     # MEASURED calibration: 245 vs output = 5 x 34 + 75
UNIF_INSTR_BLOCK = 160  # ASSUMED: the control template matches the VS template
                        # minus the probability gloss and normalization text
UNIF_JSON_OVERHEAD = 30  # ASSUMED: list-only JSON (no probabilities) vs VS's 75

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
    "improve": [
        # registry single turn, no system prompt (instruction ~48 tokens);
        # response length ASSUMED at the paraphrase band (improved text of
        # similar length).
        Stage("improve", 48, 40, "assumed"),
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
    "improve": "sentence",
    "take_inspiration": "sentence",
    "expand+compress": "long",
}

# --- prices -------------------------------------------------------------------
# gpt-6-luna published rate-card prices (VERIFIED 2026-10-31 against several
# trackers: $0.10/$0.50 list, batch $0.05/$0.25); the Azure portal parity check
# is still pending, so the constants below remain until that check.
LUNA_IN_PRICE = 0.10    # USD per M tokens, list (OpenAI rate card)
LUNA_OUT_PRICE = 0.50   # USD per M tokens, list (OpenAI rate card)
BATCH_DISCOUNT = 0.50   # ASSUMED; verify the Azure global-batch discount
USD_PER_EUR = 0.92      # ASSUMED conversion

# --- model ----------------------------------------------------------------------
def stage_call_tokens(stage: Stage, band: str, condition: str) -> tuple[int, int]:
    """(input, output) tokens for ONE call of this stage under this condition.

    direct: plain task prompt, single response. uniform_list: one joint call
    requesting k distinct candidates in a list-only JSON (input block = VS's
    minus the probability boilerplate; output = k candidate texts plus a
    smaller JSON overhead). vs_weighted: the measured VS call (input block
    with probability spec; output = k candidate texts plus the measured JSON
    overhead).
    """
    seed = SEED_TOKENS[band]
    if condition == "direct":
        return stage.instr_tokens + seed, stage.out_tokens
    if condition == "uniform_list":
        return (UNIF_INSTR_BLOCK + seed,
                stage.out_tokens * VS_K + UNIF_JSON_OVERHEAD)
    if condition == "vs_weighted":
        return (VS_INSTR_BLOCK + seed,
                stage.out_tokens * VS_K + JSON_OVERHEAD)
    raise ValueError(f"unknown condition: {condition}")


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
          f"prompt calls {cl*PANEL_MODELS:,}. Only gpt-6-luna is billed; the "
          f"open-weight models run on the Vienna cluster.")
    print(f"gpt-6-luna list (rate card): {usd:,.0f} USD ~ "
          f"{usd*USD_PER_EUR:,.0f} EUR; with batch x{BATCH_DISCOUNT}: "
          f"{usd*BATCH_DISCOUNT*USD_PER_EUR:,.0f} EUR (Azure portal parity "
          f"pending; discount ASSUMED, verify with Azure).")
    print(f"\nCost share expand+compress: "
          f"{sum(r['usd'] for r in rows if r['task']=='expand+compress')/usd:.0%} "
          f"(dominated by ASSUMED long-text VS response lengths).")

    # Locked-recount checks (handoff 2026-10-31): 1,080,000 transmitted and
    # 1,350,000 prompt calls per model for the 3-condition 4-task design
    # (3 x 4 x 90 x 5 x 200 texts; one call per stage in every condition).
    assert tr == 1_080_000, tr
    assert cl == 1_350_000, cl
    print("\nrecount checks passed: transmitted 1,080,000; prompt calls 1,350,000 (per model)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()