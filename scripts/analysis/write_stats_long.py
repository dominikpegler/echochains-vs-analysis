import os
import re
import sys
from pathlib import Path

import pandas as pd

from echochain.utils import get_data_dir

SUB_PATH = sys.argv[1]  # vs_weighted | vs_argmax

DATA_DIR = get_data_dir()
STATS_DIR = Path(DATA_DIR, "data", "echodrift_stats", SUB_PATH)
SHARD_ROOT = STATS_DIR / "shards"


def parse_config(cfg):
    m = re.search(r"temp(?P<temp>\d+(?:\.\d+)?)_?topp(?P<topp>\d+(?:\.\d+)?)", cfg)
    return float(m["temp"]), float(m["topp"])


def main():
    dfs = []
    for root, _, files in os.walk(SHARD_ROOT):
        for fn in files:
            if fn.endswith(".parquet"):
                df = pd.read_parquet(Path(root) / fn)
                df[["temperature", "top_p"]] = df["config"].apply(
                    lambda c: pd.Series(parse_config(c))
                )
                dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print("total rows:", len(df_all), flush=True)

    df_cum = df_all[df_all["kind"] == "cumulative"].copy()
    df_loc = df_all[df_all["kind"] == "local"].copy()

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    df_cum.to_csv(STATS_DIR / "metrics_cumulative_long.csv", index=False)
    df_loc.to_csv(STATS_DIR / "metrics_local_long.csv", index=False)

    idx_cum = df_cum.groupby(["model", "config", "msg_id", "chain", "metric"])["hop"].idxmax()
    df_last_cum = df_cum.loc[idx_cum].copy()
    df_last_cum.to_csv(STATS_DIR / "metrics_last_hop_cumulative.csv", index=False)

    idx_loc = df_loc.groupby(["model", "config", "msg_id", "chain", "metric"])["hop"].idxmax()
    df_last_loc = df_loc.loc[idx_loc].copy()
    df_last_loc.to_csv(STATS_DIR / "metrics_last_hop_local.csv", index=False)

    print("=== CSVs written ===", flush=True)


if __name__ == "__main__":
    main()
