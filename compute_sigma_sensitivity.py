"""Sensitivity check: recompute Sigma with poison-affected seed x config cells dropped.

The shard pipeline marked 21 (seed, config) chains as .poison (deterministic
native crashes). Each poisoned chain removes one of the 9 chains of its
(msg_id, config) cell; Sigma (between-chain variance, ddof=1) is still defined
with 8 chains, but the affected cells are the only cells with incomplete
within-cell coverage.

This script recomputes the composite Sigma tables with every affected
(msg_id, config) cell DROPPED entirely (strictest check) and compares the
headline contrasts to the full-data run from compute_sigma_exploratory.py.

Usage:
    python compute_sigma_sensitivity.py            # drop affected cells
    python compute_sigma_sensitivity.py --list     # just list affected cells
"""

import sys
from pathlib import Path

import pandas as pd

from echochain.diffusion import AXES, sigma_axis_displacement, sigma_axis_position
from echochain.utils import get_data_dir

HOP = 200
ARMS = ["direct", "vs_weighted", "vs_argmax"]

DATA_DIR = get_data_dir()
BASE = Path(DATA_DIR, "data", "echodrift_stats")

# Poison inventory (from remote .poison markers, 2026-09-03; 15 weighted + 6 argmax)
POISON = {
    ("vs_weighted", "temp1.0_topp0.3"): [("MSG_040", 1), ("MSG_050", 6)],
    ("vs_weighted", "temp1.0_topp0.5"): [
        ("MSG_033", 4), ("MSG_033", 6), ("MSG_033", 8),
        ("MSG_036", 1), ("MSG_036", 5), ("MSG_045", 2),
    ],
    ("vs_weighted", "temp1.0_topp0.8"): [
        ("MSG_027", 7), ("MSG_028", 5), ("MSG_029", 1),
        ("MSG_030", 7), ("MSG_037", 9), ("MSG_042", 9), ("MSG_044", 4),
    ],
    ("vs_argmax", "temp1.0_topp0.3"): [
        ("MSG_010", 2), ("MSG_016", 6), ("MSG_024", 8),
    ],
    ("vs_argmax", "temp1.0_topp0.5"): [
        ("MSG_015", 9), ("MSG_016", 6), ("MSG_022", 7),
    ],
}


def poisoned_cells():
    """Set of (arm, config, msg_id) cells with at least one poisoned chain."""
    return {(arm, config, msg) for (arm, config), chains in POISON.items() for msg, _ in chains}


def stats_dir(arm):
    return BASE if arm == "direct" else BASE / arm


def main():
    if "--list" in sys.argv:
        for (arm, config), chains in sorted(POISON.items()):
            for msg, ch in chains:
                print(f"{arm:12s} {config} {msg} chain_{ch:03d}")
        return

    drop = poisoned_cells()
    print(f"{len(drop)} affected (arm, config, msg_id) cells; {sum(len(v) for v in POISON.values())} poisoned chains")

    headline = {}
    for arm in ARMS:
        sd = stats_dir(arm)
        df_axis = pd.read_csv(sd / "axis_cumulative_long.csv")
        keep = pd.Series(
            [
                (arm, cfg, msg) not in drop
                for cfg, msg in zip(df_axis["config"], df_axis["msg_id"])
            ]
        )
        n_before = df_axis.groupby(["config", "msg_id"]).ngroups
        df_axis = df_axis[keep]
        n_after = df_axis.groupby(["config", "msg_id"]).ngroups

        pos = sigma_axis_position(df_axis, HOP)
        disp = sigma_axis_displacement(df_axis, HOP)
        pos["composite"] = pos[AXES].mean(axis=1)
        disp["composite"] = disp[AXES].mean(axis=1)

        for name, df in [("position", pos), ("displacement", disp)]:
            headline[(arm, name)] = df["composite"].mean()
        print(f"{arm}: cells {n_before} -> {n_after} after drop")

    print("\n===== Sensitivity: composite Sigma, affected cells dropped =====")
    rows = []
    for name in ["position", "displacement"]:
        rows.append(
            {
                "metric": name,
                "direct": headline[("direct", name)],
                "vs_weighted": headline[("vs_weighted", name)],
                "vs_argmax": headline[("vs_argmax", name)],
                "w_minus_d": headline[("vs_weighted", name)] - headline[("direct", name)],
                "a_minus_d": headline[("vs_argmax", name)] - headline[("direct", name)],
                "w_over_d": headline[("vs_weighted", name)] / max(headline[("direct", name)], 1e-12),
                "w_over_a": headline[("vs_weighted", name)] / max(headline[("vs_argmax", name)], 1e-12),
            }
        )
    out = pd.DataFrame(rows)
    print(out.round(6).to_string(index=False))

    ref = pd.read_csv(BASE / "sigma_exploratory" / "sigma_headline.csv")
    print("\n===== Full-data reference (sigma_headline.csv) =====")
    print(ref.round(6).to_string(index=False))

    out_dir = BASE / "sigma_exploratory"
    out.to_csv(out_dir / "sigma_sensitivity_drop_poison_cells.csv", index=False)
    print(f"\nwrote {out_dir / 'sigma_sensitivity_drop_poison_cells.csv'}")


if __name__ == "__main__":
    main()