"""Compute Sigma (between-chain variance) summaries for the VS paper.

Usage:
    python compute_sigma.py direct            # base (Direct) stats dir
    python compute_sigma.py vs_weighted
    python compute_sigma.py vs_argmax

For each arm it reads STATS_DIR/metrics_cumulative_long.csv and
STATS_DIR/axis_cumulative_long.csv and writes three files:
    sigma_at_hop200.csv           (a) axis score position at hop 200
    sigma_displacement_hop200.csv (b) cumulative axis displacement
    sigma_cosine_hop200.csv       (c) cosine to seed at hop 200
"""

import sys
from pathlib import Path

import pandas as pd

from echochain.diffusion import (
    AXES,
    sigma_axis_displacement,
    sigma_axis_position,
    sigma_cosine_to_seed,
)
from echochain.utils import get_data_dir

ARM = sys.argv[1] if len(sys.argv) > 1 else "direct"
HOP = 200

DATA_DIR = get_data_dir()
if ARM == "direct":
    STATS_DIR = Path(DATA_DIR, "data", "echodrift_stats")
else:
    STATS_DIR = Path(DATA_DIR, "data", "echodrift_stats", ARM)

AXIS_CSV = STATS_DIR / "axis_cumulative_long.csv"
METRICS_CSV = STATS_DIR / "metrics_cumulative_long.csv"

df_axis = pd.read_csv(AXIS_CSV)
df_metrics = pd.read_csv(METRICS_CSV)

pos = sigma_axis_position(df_axis, HOP)
disp = sigma_axis_displacement(df_axis, HOP)
cos = sigma_cosine_to_seed(df_metrics, HOP)

cols = ["model", "config", "msg_id", "hop", "metric"] + AXES
pos[cols].to_csv(STATS_DIR / "sigma_at_hop200.csv", index=False)
disp[cols].to_csv(STATS_DIR / "sigma_displacement_hop200.csv", index=False)
cos.to_csv(STATS_DIR / "sigma_cosine_hop200.csv", index=False)

n_seeds = pos["msg_id"].nunique()
print(f"{ARM}: wrote sigma CSVs; {n_seeds} seeds x {pos['config'].nunique()} configs", flush=True)
