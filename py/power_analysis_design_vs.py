"""Design power analysis for the VS confirmatory study: seeds vs chains.

Question (2026-10-30 lock): at a constant chain budget (~450 transmitted
chains per condition per task), should the design spend on more seeds with
fewer chains, or keep 50 seeds x 9 chains?

Monte Carlo over design variants under the exploratory pilot's
distribution, two source layers:

1. Seed population (EMPIRICAL): per-seed, per-axis Sigma values from the
   pilot's sigma_at_hop200.csv (50 seeds x 9 configs x 5 axes, averaged over
   configs). Designs with more than 50 seeds bootstrap seed rows with
   replacement, so larger-seed designs sample the same population harder.
2. Chain-sampling noise (ANALYTIC): a cell with C chains estimates the cell
   variance as sigma^2 x chi^2(df=C-1)/(C-1); the simulation draws exactly
   that, paired across arms seed-by-seed (the same seed text runs under both
   conditions in the real design, so its heterogeneity cancels in the
   contrast).

For each budget-neutral design (seeds x chains per condition) the script
reports:
- MDE of the headline contrast (mean across seeds and axes of per-cell
  sample-variance differences, VS-weighted minus Direct), in axis units and
  relative to the Direct mean, and restricted to the smallest genre cell
  (three genres, equal proportion).
- MDE of the per-axis mu contrast on a target axis (ASSUMED within-chain
  noise: resid_frac of the endpoint variance, pending a real residual
  decomposition from the pilot logs).
- Recovery of seed-level Sigma under VS (Spearman between true and
  estimated per-seed sigma across seeds, axes-averaged): the diagnostic
  behind widening-seed analyses, which degrades as chains per cell shrink.

Estimator note: the headline analysis averages per-cell variance estimates;
the simulated statistic matches the pilot's mean between-chain variance
construction. Caveat: parameters are paraphrase-task pilot values, so the
other tasks are extrapolations; hop trajectories are not simulated (hop-200
scores only).

Run: python py/power_analysis_design_vs.py [--reps 2000]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    os.environ.get("DATA_DIR", str(Path.home() / "Dropbox/data_export/echo-chains"))
)
AXES = [
    "valence_neg_to_pos",
    "tone_neutral_to_intense",
    "moderate_to_extreme",
    "factual_to_narrative",
    "abstract_to_concrete",
]
H = 200                # hops
GENRES = 3             # genre cells in the seed corpus (equal proportion)
ALPHA, POWER_Z = 0.05, 1.96 + 0.84   # two-sided alpha=.05, 80% power
ARMS = {"direct": "direct", "vs": "vs_weighted"}
DESIGNS = {"50x9": (50, 9), "75x6": (75, 6), "112x4": (112, 4), "150x3": (150, 3)}
MU_TARGET_AXIS = "abstract_to_concrete"   # the pilot's strongest drift axis
RESID_FRAC = 0.5    # ASSUMED share of endpoint variance that is within-chain noise
N_HOPS_SLOPE = 200  # chains' hop series feeding the per-chain OLS slope


def load_seed_population():
    """Per-seed x per-axis sigma (config-averaged), per arm.

    The direct arm's sigma table sits at echodrift_stats/ root; the VS arms
    have their own subdirectories (historical layout).
    """
    pop = {}
    for arm in ARMS.values():
        base = Path(DATA_DIR) / "data" / "echodrift_stats"
        fp = (base / arm / "sigma_at_hop200.csv" if arm != "direct"
              else base / "sigma_at_hop200.csv")
        sd = pd.read_csv(fp)
        per_seed = sd.groupby("msg_id")[AXES].mean()   # average configs
        pop[arm] = per_seed[AXES].to_numpy()           # (50 seeds, 5 axes)
    return pop


def load_mu_params():
    fp = Path(DATA_DIR) / "pub" / "descriptives_vs" / "metrics_pub.json"
    d = json.loads(fp.read_text())
    mu = {arm: d["mu"][arm][MU_TARGET_AXIS] for arm in ARMS.values()}
    sigma = {arm: d["sigma"]["per_axis"][arm][MU_TARGET_AXIS] for arm in ARMS.values()}
    return mu, sigma


def simulate_design(pop, mu, sigma_mu, seeds, chains, reps, rng):
    axes_n = len(AXES)
    genre_sizes = [seeds // GENRES] * GENRES
    genre_sizes[-1] += seeds - sum(genre_sizes[:-1])
    overall, genre = [], [[] for _ in range(GENRES)]
    rhos, mu_hat_diff = [], []
    n_pop = pop["direct"].shape[0]
    for _ in range(reps):
        idx = rng.integers(0, n_pop, size=seeds)   # bootstrap seed rows
        cell_var, true_sig = {}, {}
        for arm in ARMS.values():
            true = pop[arm][idx]                 # (seeds, axes) true sigma
            chi = rng.chisquare(chains - 1, size=(seeds, axes_n)) / (chains - 1)
            cell_var[arm] = true**2 * chi        # per-cell sample variances
            true_sig[arm] = true
        diff = cell_var["vs_weighted"] - cell_var["direct"]
        overall.append(diff.mean())
        start = 0
        for g, gsize in enumerate(genre_sizes):
            genre[g].append(diff[start : start + gsize].mean())
            start += gsize
        # per-seed sigma recovery (VS arm, axes-averaged)
        est = np.sqrt((cell_var["vs_weighted"]).mean(axis=1))
        tru = true_sig["vs_weighted"].mean(axis=1)
        rhos.append(pd.Series(est).rank().corr(pd.Series(tru).rank()))
        # mu contrast on the target axis: per-chain slope estimates
        denom = math.sqrt(N_HOPS_SLOPE) * N_HOPS_SLOPE / math.sqrt(12)
        se_mu = math.sqrt(RESID_FRAC * sigma_mu["vs_weighted"]) / denom
        se_mu_d = math.sqrt(RESID_FRAC * sigma_mu["direct"]) / denom
        mu_vs = rng.normal(mu["vs_weighted"], se_mu, size=seeds * chains)
        mu_dir = rng.normal(mu["direct"], se_mu_d, size=seeds * chains)
        mu_hat_diff.append(mu_vs.mean() - mu_dir.mean())
    direct_mean = float(pop["direct"].mean())
    true_delta = float((pop["vs_weighted"] - pop["direct"]).mean())
    rows = [("overall", POWER_Z * np.std(overall, ddof=1))]
    for g in range(GENRES):
        rows.append((f"genre{g}", POWER_Z * np.std(genre[g], ddof=1)))
    return dict(
        rows=rows,
        mu_mde=POWER_Z * float(np.std(mu_hat_diff, ddof=1)),
        spearman=float(np.nanmean(rhos)),
        direct_mean=direct_mean,
        true_delta=true_delta,
        true_mu_delta=mu["vs_weighted"] - mu["direct"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    pop = load_seed_population()
    mu, sigma_mu = load_mu_params()
    spread = np.exp(np.std(np.log(np.maximum(pop["vs_weighted"], 1e-6)), axis=0))
    print(f"pilot seed population: {pop['direct'].shape[0]} seeds x {len(AXES)} axes")
    print(f"empirical per-axis seed spread (exp of log-sigma SD): "
          f"{ {a: round(float(s), 1) for a, s in zip(AXES, spread)} }")
    print(f"true pilot contrast (seed-averaged, VS minus Direct): "
          f"{(pop['vs_weighted'] - pop['direct']).mean():.2e} axis units; "
          f"Direct mean {pop['direct'].mean():.2e}\n")
    print(f"simulating {args.reps} replications per design\n")
    header = (f"{'design':<8}{'MDE mean-Sigma':>16}{'rel. to Direct':>15}"
              f"{'MDE genre cell':>16}{'rel. to Direct':>15}"
              f"{'MDE mu contrast':>17}{'Spearman':>10}")
    print(header)
    print("-" * len(header))
    for name, (s, c) in DESIGNS.items():
        rng = np.random.default_rng(42)
        d = simulate_design(pop, mu, sigma_mu, s, c, args.reps, rng)
        overall = d["rows"][0]
        smallest = min(d["rows"][1:], key=lambda r: r[1])
        print(f"{name:<8}{overall[1]:>16.2e}{overall[1]/d['direct_mean']:>15.1%}"
              f"{smallest[1]:>16.2e}{smallest[1]/d['direct_mean']:>15.1%}"
              f"{d['mu_mde']:>17.2e}{d['spearman']:>10.2f}")
    print("-" * len(header))
    print(f"mu column: MDE on '{MU_TARGET_AXIS}' (VS minus Direct); true pilot "
          f"delta {d['true_mu_delta']:.2e}; ASSUMED resid_frac={RESID_FRAC} "
          f"(within-chain noise share), so treat mu MDEs as indicative.")
    print("\nreading guide: MDE = smallest true VS-minus-Direct difference the "
          "design detects with 80% power at alpha=.05 (two-sided); smaller is "
          "better. The true mean-Sigma delta is ~7.5x the Direct mean, so the "
          "headline contrast is comfortable everywhere; the genre cell and "
          "Spearman (per-seed Sigma recovery) discriminate the designs.")
    print("NOTE: paraphrase-task pilot parameters; other tasks are extrapolations. "
          "Designs with >50 seeds bootstrap the 50-seed pilot population.")


if __name__ == "__main__":
    main()