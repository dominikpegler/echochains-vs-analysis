"""Exploratory Sigma summary for the gpt-4.1-nano pilot (VS vs Direct).

Reads the completed stats/axis CSVs for the three arms (direct, vs_weighted,
vs_argmax) and writes:
  - a composite Sigma table (mean + per-axis between-chain variance at hop 200)
  - the H4-direction contrast (VS-weighted vs VS-argmax on composite Sigma)
  - the H1/H3-direction context (Direct baseline, VS-weighted inflation)

Intended to be re-run after rsync; reads from the local DATA_DIR by default.
"""

from pathlib import Path

import pandas as pd

from echochain.diffusion import (
    AXES,
    sigma_axis_displacement,
    sigma_axis_position,
    sigma_cosine_to_seed,
)
from echochain.utils import get_data_dir

HOP = 200
ARMS = ["direct", "vs_weighted", "vs_argmax"]

DATA_DIR = get_data_dir()
BASE = Path(DATA_DIR, "data", "echodrift_stats")


def stats_dir(arm):
    return BASE if arm == "direct" else BASE / arm


def composite(row):
    """Mean across the 5 axes of a sigma row."""
    return row[AXES].mean()


def load_sigma(arm):
    sd = stats_dir(arm)
    pos = sigma_axis_position(pd.read_csv(sd / "axis_cumulative_long.csv"), HOP)
    disp = sigma_axis_displacement(pd.read_csv(sd / "axis_cumulative_long.csv"), HOP)
    cos = sigma_cosine_to_seed(pd.read_csv(sd / "metrics_cumulative_long.csv"), HOP)
    pos["composite"] = pos[AXES].apply(composite, axis=1)
    disp["composite"] = disp[AXES].apply(composite, axis=1)
    return pos, disp, cos


def summarize(arm):
    pos, disp, cos = load_sigma(arm)
    out = {}
    cols = AXES + ["composite"]
    for name, df in [("position", pos), ("displacement", disp)]:
        grp = df.groupby("config")[cols].mean()
        grp["arm"] = arm
        out[name] = grp.reset_index().set_index(["arm", "config"])
    cg = cos.groupby("config")["value"].mean().rename("cosine_sigma").to_frame()
    cg["arm"] = arm
    cg = cg.reset_index().set_index(["arm", "config"])
    return out


def main():
    for name in ["position", "displacement"]:
        tbl = pd.concat(summarize(a)[name] for a in ARMS)
        per = tbl.reset_index().pivot_table(index="config", columns="arm", values="composite")
        per = per.reindex(columns=ARMS)
        per["vs_weighted_minus_direct"] = per["vs_weighted"] - per["direct"]
        per["vs_argmax_minus_direct"] = per["vs_argmax"] - per["direct"]
        per["weighted_minus_argmax"] = per["vs_weighted"] - per["vs_argmax"]
        print(f"\n===== Sigma composite ({name}, hop {HOP}) =====")
        print(per.round(6).to_string())
        out = Path(DATA_DIR, "data", "echodrift_stats", "sigma_exploratory")
        out.mkdir(parents=True, exist_ok=True)
        per.to_csv(out / f"sigma_composite_{name}_hop{HOP}.csv")

    # Aggregate across configs for the headline contrast
    rows = []
    for arm in ARMS:
        for name in ["position", "displacement"]:
            df = summarize(arm)[name]
            rows.append(
                {
                    "arm": arm,
                    "metric": name,
                    "mean_composite_sigma": df["composite"].mean(),
                    "weighted_difference_to_direct": None,
                }
            )
    agg = pd.DataFrame(rows)
    agg.loc[agg["arm"] == "vs_weighted", "weighted_difference_to_direct"] = (
        agg.loc[agg["arm"] == "vs_weighted", "mean_composite_sigma"].values[0]
        - agg.loc[agg["arm"] == "direct", "mean_composite_sigma"].values[0]
    )
    agg.loc[agg["arm"] == "vs_argmax", "weighted_difference_to_direct"] = (
        agg.loc[agg["arm"] == "vs_argmax", "mean_composite_sigma"].values[0]
        - agg.loc[agg["arm"] == "direct", "mean_composite_sigma"].values[0]
    )
    wgt = agg.loc[agg["arm"] == "vs_weighted", "mean_composite_sigma"].values[0]
    amx = agg.loc[agg["arm"] == "vs_argmax", "mean_composite_sigma"].values[0]
    direct = agg.loc[agg["arm"] == "direct", "mean_composite_sigma"].values[0]
    print("\n===== Headline (mean across 9 configs, hop 200) =====")
    print(agg.drop(columns=["weighted_difference_to_direct"]).round(6).to_string(index=False))
    print(f"\nDirect vs_weighted sigma ratio: {wgt / max(direct, 1e-12):.4f}")
    print(f"VS-weighted vs VS-argmax sigma ratio: {wgt / max(amx, 1e-12):.4f}  (VS-weighted > argmax => H4 direction)")
    agg.to_csv(Path(DATA_DIR, "data", "echodrift_stats", "sigma_exploratory", "sigma_headline.csv"), index=False)
    print("\nwrote sigma_exploratory/*.csv")


if __name__ == "__main__":
    main()
