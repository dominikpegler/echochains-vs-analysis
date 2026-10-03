#!/usr/bin/env python3
"""Staged publication-output builder for the VS paper (figures, tables, JSON).

Restructure of the output half of py/analysis_01_descriptives_vs.py
(2026-10-03): color/label/geometry changes must not require reprocessing
the full study. The merged notebook stays the entry point for a complete
re-analysis; this script regenerates only the publication outputs it
needs, in independent stages, from the published CSV exports (no LM
calls, no refitting).

The one measured bottleneck of the monolithic run is drift_slopes: the
groupby-apply polyfit loop (12,150 chain groups per condition x 5 axes)
takes ~1 h/condition, while the closed-form vectorized slope is 0.4 s
and verified numerically identical on the real exports.

Stages (each runnable alone; all read-only w.r.t. the data exports):

  tables    last-hop tables, combined last-hop, ICC tables,
            metrics_pub.json, tab_sigma_headline / per_axis /
            mu_per_axis / sensitivity
  figures   fig_pilot_cum_{surface,factual,semantic}, fig_info_core,
            fig_info_entailment, fig_sigma_headline,
            fig_sigma_per_axis_curves, fig_decoding_sensitivity
            (paper geometry: 184-193 mm grids, base_font 7, 600 dpi PNG)
  examples  the manuscript's exploratory.example texts + sigma/mu macro
            sections merged into the three metrics_pub.json files

Outputs land in DATA/pub/descriptives{,_vs,_vs_argmax}/ as before, so
the manuscript's figure paths and org macros are unchanged.

Run (echochain env, from the repo root):
  conda run -n echochain python scripts/make_paper_outputs.py --stage all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in [REPO, REPO / "pub-utils"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from echochain.diffusion import AXES, sigma_axis_position
from echochain.utils import get_data_dir
from echochain.icc_utils import compute_icc
from echochain.plot_utils import ci95
from echochain.constants import CONDITION_COLORS, CONDITION_LABELS

import pub_utils.figures as F
from pub_utils.tables import TableSpec, write_table

DATA_DIR = get_data_dir()
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")
STATS_BASE = Path(DATA_DIR, "data", "echodrift_stats")
PUB_BASE = Path(DATA_DIR, "pub")
MSG_FILE = REPO / "seed_sentences_50.json"

HOP = 200
CONFIG = "temp0.8_topp0.5"
METRICS_VERSION = "v1.0.0"

CONDITIONS = ["direct", "vs_argmax", "vs_weighted"]
STACKED_CONDITIONS = CONDITIONS

CONDITION_CONFIG = {
    "direct": dict(
        analysis_name="descriptives",
        sub_path=None,
        condition_label="direct prompting",
        chosen_models=None,
        chosen_configs=None,
    ),
    "vs_weighted": dict(
        analysis_name="descriptives_vs",
        sub_path="vs_weighted",
        condition_label="verbalized sampling (weighted)",
        chosen_models=["gpt-4.1-nano"],
        chosen_configs=["temp0.8_topp0.5"],
    ),
    "vs_argmax": dict(
        analysis_name="descriptives_vs_argmax",
        sub_path="vs_argmax",
        condition_label="verbalized sampling (mode)",
        chosen_models=["gpt-4.1-nano"],
        chosen_configs=["temp0.8_topp0.5"],
    ),
}

AXIS_LABELS = {
    "valence_neg_to_pos": "Valence",
    "tone_neutral_to_intense": "Tone intensity",
    "moderate_to_extreme": "Extremity",
    "factual_to_narrative": "Factual vs narrative",
    "abstract_to_concrete": "Abstract vs concrete",
}

TITLES = {
    "delta_len": r"$\Delta$ tokens",
    "ratio_len": "Expansion ratio",
    "jaccard": "Lexical Jaccard",
    "entity_f1": "Entity F1",
    "triple_f1": "Triple F1",
    "cosine": "Embedding cosine",
    "entail_ab": "Entail seed→hop",
    "entail_ba": "Entail hop→seed",
    "entail_consistent": "Entailment consistency",
}

KEY_METRICS = [
    "delta_len", "ratio_len", "jaccard", "entity_f1", "triple_f1",
    "cosine", "entail_ab", "entail_ba", "entail_consistent", "k_unique",
]
OVERLAY_METRICS = ["cosine", "jaccard"]
ENTAILMENT_METRICS = ["entail_ab", "entail_ba", "entail_consistent"]
OVERLAY_PLOT_METRICS = OVERLAY_METRICS + ENTAILMENT_METRICS
OVERLAY_TITLES = {
    "cosine": "Cosine similarity to seed",
    "jaccard": "Lexical Jaccard",
}
ENTAILMENT_TITLES = {
    "entail_ab": "Seed entails hop",
    "entail_ba": "Hop entails seed",
    "entail_consistent": "Mutual entailment",
}
ANCHOR_VALUES = {
    "delta_len": 0.0, "ratio_len": 1.0, "jaccard": 1.0, "entity_f1": 1.0,
    "triple_f1": 1.0, "cosine": 1.0, "entail_ab": 1.0, "entail_ba": 1.0,
    "entail_consistent": 1.0,
}
STACKED_LAYER_SPECS = {
    "surface": (["delta_len", "ratio_len", "jaccard"], [False, False, True]),
    "factual": (["entity_f1", "triple_f1"], [True, True]),
    "semantic": (
        ["cosine", "entail_ab", "entail_ba", "entail_consistent"],
        [True, True, True, True],
    ),
}
STACKED_METRICS = [
    "delta_len", "ratio_len", "jaccard", "entity_f1", "triple_f1",
    "cosine", "entail_ab", "entail_ba", "entail_consistent",
]

MANIFEST = PUB_BASE / "descriptives_vs" / "outputs_manifest.json"


def stats_dir(condition):
    return STATS_BASE if condition == "direct" else STATS_BASE / condition


# --- vectorized cores ------------------------------------------------------


def drift_slopes(df_axis):
    """Per-chain OLS slope of each axis on hop, closed-form, vectorized.

    Numerically identical to np.polyfit(h, y, 1)[0] per group (verified
    on the published exports); replaces the groupby-apply polyfit loop
    that dominated the monolithic run time.
    """
    keys = ["config", "msg_id", "chain"]
    grp = df_axis.groupby(keys, sort=True)
    ybar = grp[AXES].transform("mean").to_numpy(float)
    hbar = grp["hop"].transform("mean").to_numpy(float)[:, None]
    h = df_axis["hop"].to_numpy(float)[:, None]
    hd = (h - hbar).ravel()
    yd = df_axis[AXES].to_numpy(float) - ybar
    num = pd.DataFrame(hd[:, None] * yd, columns=AXES)
    for k in keys:
        num[k] = df_axis[k].to_numpy()
    num = num.groupby(keys, sort=True).sum()
    den = pd.DataFrame({"hd2": hd * hd})
    for k in keys:
        den[k] = df_axis[k].to_numpy()
    den = den.groupby(keys, sort=True)["hd2"].sum()
    return num.div(den, axis=0).reset_index()


def drift_summary(df_axis):
    slopes = drift_slopes(df_axis)
    mu = slopes[AXES].mean()
    mu["mean_abs"] = mu.abs().mean()
    return mu


def sigma_curve_position(df_axis):
    var = df_axis.groupby(["config", "msg_id", "hop"])[AXES].var(ddof=1)
    var["mean_sigma"] = var[AXES].mean(axis=1)
    return var.groupby("hop")["mean_sigma"].mean().reset_index()


def sigma_curve_position_per_axis(df_axis):
    var = df_axis.groupby(["config", "msg_id", "hop"])[AXES].var(ddof=1)
    return var.groupby("hop")[AXES].mean().reset_index()


def load_axis(condition):
    # full file: sigma_axis_position groups on ["model", "config", "msg_id"],
    # so the "model" column must be present (my usecols drop broke it).
    return pd.read_csv(stats_dir(condition) / "axis_cumulative_long.csv")


def load_metrics(condition, config=None):
    cols = ["model", "config", "msg_id", "chain", "metric", "hop", "value"]
    df = pd.read_csv(stats_dir(condition) / "metrics_cumulative_long.csv",
                     usecols=cols)
    if config is not None:
        df = df[df["config"] == config]
    return df


def last_hop_idx(d):
    return d.groupby(["msg_id", "chain"])["hop"].idxmax()


def last_hop_series(df_one_cfg, metric):
    d = df_one_cfg[df_one_cfg["metric"] == metric]
    if d.empty:
        return pd.DataFrame(columns=["msg_id", "chain", "value"])
    return d.loc[last_hop_idx(d), ["msg_id", "chain", "value"]].copy()


def metric_mean_ci_at_last(df, metric):
    idx = last_hop_idx(df[df["metric"] == metric])
    vals = df.loc[idx, "value"].values
    mu, h = ci95(vals)
    return dict(mean=float(mu), ci_lo=float(mu - h), ci_hi=float(mu + h),
                N=int(len(vals)))


def sample_sizes(df):
    return {
        "n_seeds": int(df["msg_id"].nunique()),
        "n_chains": int(df.groupby(["msg_id", "chain"]).ngroups),
        "n_obs": int(len(df)),
    }


def json_safe_key(cfg):
    return str(cfg).replace(".", "p").replace(",", "_").replace(" ", "")


# --- tables stage ----------------------------------------------------------


def last_hop_summary(df_one_cfg):
    idx = last_hop_idx(df_one_cfg)
    d = df_one_cfg.loc[idx].copy()
    rows = []
    for m, g in d.groupby("metric"):
        mean, h95 = ci95(g["value"].values)
        rows.append(dict(metric=m, mean=mean, ci_lo=mean - h95,
                         ci_hi=mean + h95, N=len(g)))
    return (pd.DataFrame(rows).set_index("metric")
            .reindex(KEY_METRICS).reset_index())


def _lasthop_spec_head():
    return dict(
        wrap="threeparttable", width="2col", use_tabularx=False,
        use_siunitx=True, float_fmt="{:.3f}",
        header_overrides={
            "entail_ab": "entail seed→hop",
            "entail_ba": "entail hop→seed",
            "entail_consistent": "entail consistent",
        },
        numeric_as_r={"N"},
    )


def write_last_hop_table_for_config(df_cum, model, config, pub_dir,
                                    condition_label, condition):
    tag = f"{model}_{config}".replace(".", "p")
    df_one = df_cum[(df_cum["model"] == model) & (df_cum["config"] == config)]
    tbl = last_hop_summary(df_one)
    disp = tbl[["metric", "mean", "ci_lo", "ci_hi", "N"]].copy()
    disp.columns = ["Metric", "Mean", "CI low", "CI high", "N"]
    disp["Metric"] = disp["Metric"].str.replace("_", " ")
    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} ({config.replace("_", " ")}) — {condition_label} — Last-hop cumulative""",
        label=f"tab:lasthop_{tag}_{condition}",
        placement="htbp", fontsize_pt=(10, 12), **_lasthop_spec_head(),
    )
    out = pub_dir / f"tab_lasthop_{tag}.tex"
    write_table(disp, out, spec)
    print("Wrote", out)


def write_combined_last_hop_table(df_cum, model, configs, pub_dir,
                                  condition_label, condition):
    fname = f"tab_lasthop_{model.replace('.', 'p')}_all"
    rows = []
    for cfg in configs:
        d = df_cum[(df_cum["model"] == model) & (df_cum["config"] == cfg)]
        s = last_hop_summary(d)
        cfg = cfg.replace("_", "/").replace("temp", "").replace("topp", "")
        s["config"] = cfg
        rows.append(s)
    big = pd.concat(rows, ignore_index=True)
    wide = big.pivot(index="metric", columns="config")[
        ["mean", "ci_lo", "ci_hi", "N"]].reindex(KEY_METRICS)
    wide.index = wide.index.str.replace("_", " ")
    wide = wide.iloc[:, :9]  # mean only (as in the notebook)
    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} — {condition_label} — Last-hop cumulative across configs (temperature / top-p)""",
        label=f"tab:lasthop_{model.replace('.', 'p')}_all_{condition}",
        width="2col", wrap="threeparttable", use_tabularx=False,
        placement="tbp", use_siunitx=False, float_fmt="{:.3f}",
        fontsize_pt=(9, 11), use_index=True,
    )
    out = pub_dir / f"{fname}.tex"
    write_table(wide, out, spec)
    print("Wrote", out)


def icc_for_config(df_cum, model, config):
    rows = []
    dcfg = df_cum[(df_cum["model"] == model) & (df_cum["config"] == config)]
    eps = 1e-6
    s_cos = last_hop_series(dcfg, "cosine")
    if not s_cos.empty:
        s_cos = s_cos.assign(
            outcome=np.log(
                (s_cos["value"].clip(eps, 1 - eps) + eps)
                / (1 - s_cos["value"].clip(eps, 1 - eps) + eps)
            )
        )
        df_for_icc = s_cos[["msg_id", "outcome"]].rename(
            columns={"outcome": "Cosine (logit)"})
        b, w, icc_val = compute_icc(df_for_icc, "Cosine (logit)", "msg_id")
        rows.append(dict(config=config, outcome="Cosine (logit)",
                         sigma2_between=float(b), sigma2_within=float(w),
                         ICC=float(icc_val),
                         N_groups=int(df_for_icc["msg_id"].nunique()),
                         N_obs=int(df_for_icc.shape[0])))
    s_dl = last_hop_series(dcfg, "delta_len")
    if not s_dl.empty:
        tmp = s_dl.rename(columns={"value": r"$\Delta$ tokens"})
        df_for_icc = tmp[["msg_id", r"$\Delta$ tokens"]]
        b, w, icc_val = compute_icc(df_for_icc, r"$\Delta$ tokens", "msg_id")
        rows.append(dict(config=config, outcome=r"$\Delta$ tokens",
                         sigma2_between=float(b), sigma2_within=float(w),
                         ICC=float(icc_val),
                         N_groups=int(df_for_icc["msg_id"].nunique()),
                         N_obs=int(df_for_icc.shape[0])))
    return rows


def widen_icc_column(body, spec_):
    return body.replace(
        r"begin{tabular}{lllS[table-format=3.3]S[table-format=2.3]S[table-format=1.3]rr}",
        r"begin{tabular}{lllS[table-format=3.3]S[table-format=2.3]S[table-format=2.3]rr}",
    )


def write_icc_table(df_cum, model, configs, pub_dir, condition_label,
                    condition):
    all_rows = []
    for cfg in configs:
        all_rows.extend(icc_for_config(df_cum, model, cfg))
    t = pd.DataFrame(all_rows)
    t["config_order"] = t["config"].astype(str)
    t = t.sort_values(["config_order", "outcome"]).drop(columns=["config_order"])
    t = t.rename(columns={
        "config": "Config", "outcome": "Outcome",
        "sigma2_between": r"$\sigma^2_{\mathrm{between}}$",
        "sigma2_within": r"$\sigma^2_{\mathrm{within}}$",
        "ICC": "ICC", "N_groups": "$N_{groups}$", "N_obs": "$N$",
    })
    t["Config"] = t["Config"].str.replace("_", " ")
    model_safe = model.replace(".", "p")
    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} — {condition_label} — Last-hop ICC by outcome and config (group = seed sentence). Outcomes computed at last hop per chain; ICC computed across messages (groups) using per-chain values.""",
        label=f"tab:icc_{model_safe}_configs_{condition}",
        wrap="threeparttable", width="2col", use_tabularx=False,
        placement="htbp", use_siunitx=True, fontsize_pt=(10, 12),
        float_fmt="{:.3f}", use_index=False,
        numeric_as_r={"$N_{groups}$", "$N$"},
        post_latex=widen_icc_column,
    )
    out = pub_dir / f"tab_icc_{model_safe}_configs.tex"
    write_table(t, out, spec)
    print("Wrote", out)


def write_metrics_pub(df_cum, models, configs, pub_dir):
    pub = {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "metrics_version": METRICS_VERSION,
            "models": models, "configs": configs,
            "builder": "scripts/make_paper_outputs.py (staged)",
        },
        "config_key_map": {}, "model_key_map": {}, "per_model": {}, "icc": {},
    }
    for model in models:
        model_safe = json_safe_key(model)
        pub["model_key_map"][model_safe] = model
        pub["per_model"][model_safe] = {}
        for cfg in configs:
            cfg_safe = json_safe_key(cfg)
            pub["config_key_map"][cfg_safe] = cfg
            d = df_cum[(df_cum["model"] == model) & (df_cum["config"] == cfg)]
            entry = {"sample": sample_sizes(d), "last_hop": {}}
            for m in KEY_METRICS:
                entry["last_hop"][m] = metric_mean_ci_at_last(d, m)
            pub["per_model"][model_safe][cfg_safe] = entry
    out = pub_dir / "metrics_pub.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pub, f, ensure_ascii=False, indent=2)
    print("Wrote", out)


def sigma_and_mu_tables(summaries):
    headline = pd.read_csv(STATS_BASE / "sigma_exploratory" / "sigma_headline.csv")
    headline = headline.rename(columns={"arm": "condition"})
    sensitivity = pd.read_csv(
        STATS_BASE / "sigma_exploratory"
        / "sigma_sensitivity_drop_poison_cells.csv")
    w = headline.loc[headline["condition"] == "vs_weighted", "mean_sigma"].values[0]
    d = headline.loc[headline["condition"] == "direct", "mean_sigma"].values[0]
    a = headline.loc[headline["condition"] == "vs_argmax", "mean_sigma"].values[0]
    print(f"headline: direct={d:.6f} weighted={w:.6f} argmax={a:.6f}")
    print(f"ratios: w/d = {w/d:.2f}; w/a = {w/a:.2f}")

    out_dir = PUB_BASE / "descriptives_vs"

    headline_tbl = pd.DataFrame({
        "Condition": [CONDITION_LABELS[c] for c in CONDITIONS],
        r"Mean $\Sigma$": [d, a, w],
        r"Ratio vs Direct": [1.0, a / d, w / d],
        r"Ratio vs VS": [d / w, a / w, 1.0],
    })
    spec = TableSpec(
        caption="Mean between-chain variance $\\Sigma$ at hop 200 (mean across 9 decoding configs and 50 seeds), gpt-4.1-nano pilot. Mean $\\Sigma$ is the mean between-chain variance across the five semantic axes; ratios are computed on this mean.",
        label="tab:sigma_headline", wrap="threeparttable", width="2col",
        use_tabularx=False, placement="htbp", use_siunitx=True,
        fontsize_pt=(9, 11), float_fmt="{:.4f}", use_index=False,
    )
    write_table(headline_tbl, out_dir / "tab_sigma_headline.tex", spec)

    per_axis = pd.DataFrame(
        {"Condition": [CONDITION_LABELS[c] for c in CONDITIONS]})
    for ax in AXES:
        per_axis[AXIS_LABELS[ax]] = [
            summaries[c]["sigma_hop200"].groupby("config")[ax].mean().mean()
            for c in CONDITIONS
        ]
    per_axis[r"Mean"] = [d, a, w]
    spec = TableSpec(
        caption="Per-axis between-chain variance $\\Sigma$ at hop 200 (mean across configs and seeds), gpt-4.1-nano pilot.",
        label="tab:sigma_per_axis", wrap="threeparttable", width="2col",
        use_tabularx=False, placement="htbp", use_siunitx=True,
        fontsize_pt=(9, 11), float_fmt="{:.4f}", use_index=False,
    )
    write_table(per_axis, out_dir / "tab_sigma_per_axis.tex", spec)

    mu_tbl = pd.DataFrame(
        {"Condition": [CONDITION_LABELS[c] for c in CONDITIONS]})
    for ax in AXES:
        mu_tbl[AXIS_LABELS[ax]] = [summaries[c]["mu"][ax] for c in CONDITIONS]
    mu_tbl[r"Mean $|\mu|$"] = [summaries[c]["mu"]["mean_abs"] for c in CONDITIONS]
    spec = TableSpec(
        caption="Per-axis drift $\\mu$ (mean ordinary least squares slope of axis score on hop, per chain), gpt-4.1-nano pilot. Mean $|\\mu|$ is the mean absolute drift across the five axes.",
        label="tab:mu_per_axis", wrap="threeparttable", width="2col",
        use_tabularx=False, placement="htbp", use_siunitx=True,
        fontsize_pt=(9, 11), float_fmt="{:.5f}", use_index=False,
    )
    write_table(mu_tbl, out_dir / "tab_mu_per_axis.tex", spec)

    sens_tbl = pd.DataFrame({
        "Metric": ["Position"],
        r"$\Sigma$ Direct": [sensitivity["direct"].values[0]],
        r"$\Sigma$ VS": [sensitivity["vs_weighted"].values[0]],
        r"$\Sigma$ Verbalized mode": [sensitivity["vs_argmax"].values[0]],
        r"Ratio weighted/direct": [sensitivity["w_over_d"].values[0]],
        r"Ratio weighted/argmax": [sensitivity["w_over_a"].values[0]],
    })
    spec = TableSpec(
        caption="Sensitivity check: mean $\\Sigma$ at hop 200 after dropping all 18 poison-affected seed x config cells. 21 of 8,100 chains (0.26\\%) were excluded as deterministically crash-poisoned; the 18 affected seed x config cells are dropped entirely here.",
        label="tab:sigma_sensitivity", wrap="threeparttable", width="2col",
        use_tabularx=False, placement="htbp", use_siunitx=True,
        fontsize_pt=(9, 11), float_fmt="{:.4f}", use_index=False,
    )
    write_table(sens_tbl, out_dir / "tab_sigma_sensitivity.tex", spec)

    print("Wrote sigma tables to", out_dir)


def run_tables():
    for condition in CONDITIONS:
        cfg = CONDITION_CONFIG[condition]
        pub_dir = PUB_BASE / cfg["analysis_name"]
        pub_dir.mkdir(parents=True, exist_ok=True)

        df_cum = load_metrics(condition)
        df_axis = load_axis(condition)

        models = (cfg["chosen_models"]
                  if cfg["chosen_models"] is not None
                  else sorted(df_cum["model"].unique().tolist()))
        configs = (cfg["chosen_configs"]
                   if cfg["chosen_configs"] is not None
                   else sorted(df_cum["config"].dropna().unique().tolist()))
        all_configs = sorted(df_cum["config"].dropna().unique().tolist())

        for model in models:
            for c in configs:
                write_last_hop_table_for_config(
                    df_cum, model, c, pub_dir, cfg["condition_label"], condition)
        for model in models:
            write_combined_last_hop_table(
                df_cum, model, all_configs, pub_dir,
                cfg["condition_label"], condition)
        for model in models:
            write_icc_table(df_cum, model, configs, pub_dir,
                            cfg["condition_label"], condition)
        write_metrics_pub(df_cum, models, configs, pub_dir)

        s = {"sigma_hop200": sigma_axis_position(df_axis, HOP)}
        s["sigma_hop200"]["mean_sigma"] = s["sigma_hop200"][AXES].mean(axis=1)
        s["mu"] = drift_summary(df_axis)
        summaries_store[condition] = s
        del df_cum, df_axis
        gc.collect()
        print(f"=== processed {condition} (tables) ===")

    sigma_and_mu_tables(summaries_store)


summaries_store = {}


# --- figures stage ---------------------------------------------------------


def _mu_by_config(slopes, configs):
    rows = []
    for cfg_, grp in slopes.groupby("config"):
        per_chain_abs = grp[AXES].abs().mean(axis=1)
        rows.append({"config": cfg_, "mean_abs": per_chain_abs.mean()})
    return (pd.DataFrame(rows).set_index("config")
            .reindex(configs)["mean_abs"].to_numpy())


def _lasthop_by_config(condition, metric, configs):
    df = pd.read_csv(stats_dir(condition) / "metrics_last_hop_cumulative.csv")
    df = df[(df["metric"] == metric) & (df["hop"] == HOP)]
    return df.groupby("config")["value"].mean().reindex(configs).to_numpy()


def plot_measure_row(axs, condition_data, conditions, measure, *,
                     bounded=False, show_title=True, show_xlabel=True):
    for ax, condition in zip(axs, conditions):
        df_layer = condition_data[condition]
        hops = sorted(df_layer["hop"].unique().tolist())
        xs, mu, lo, hi = [], [], [], []
        for h in hops:
            vals = df_layer[
                (df_layer["hop"] == h) & (df_layer["metric"] == measure)
            ]["value"].tolist()
            if not vals:
                continue
            mean, h95 = ci95(vals)
            xs.append(h)
            mu.append(mean)
            lo.append(mean - h95)
            hi.append(mean + h95)
        anchor = ANCHOR_VALUES.get(measure)
        if anchor is not None:
            xs = [0] + xs
            mu = [anchor] + mu
            lo = [anchor] + lo
            hi = [anchor] + hi
        ax.plot(xs, mu, color=CONDITION_COLORS[condition])
        ax.fill_between(xs, lo, hi, color=CONDITION_COLORS[condition], alpha=0.2)
        if show_title:
            ax.set_title(CONDITION_LABELS[condition])
        if show_xlabel:
            ax.set_xlabel("Hop")
        else:
            ax.set_xlabel("")
        if bounded:
            ax.set_ylim(0, 1)


def plot_cloud(ax, df_axis, msg_id, axis, color, label, yaxis=True, ylim=None):
    sub = df_axis[df_axis["msg_id"] == msg_id]
    for chain, g in sub.groupby("chain"):
        g = g.sort_values("hop")
        ax.plot(g["hop"], g[axis], color=color, lw=0.6, alpha=0.5)
    ax.set_xlabel("Hop")
    if yaxis:
        ax.set_ylabel(AXIS_LABELS[axis])
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(label)


def run_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F.setup_style(profile="nature", use_tex=False, base_font=7,
                  minor_ticks=False, title_font_delta=0)
    STACKED_PUB = PUB_BASE / "descriptives_vs"
    STACKED_PUB.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for condition in CONDITIONS:
        df_cum_cfg = load_metrics(condition, config=CONFIG)
        stacked = df_cum_cfg[df_cum_cfg["metric"].isin(STACKED_METRICS)][
            ["metric", "hop", "value"]].copy()
        overlay = (df_cum_cfg[df_cum_cfg["metric"].isin(OVERLAY_PLOT_METRICS)]
                   .groupby(["metric", "hop"])["value"].mean().reset_index())
        del df_cum_cfg
        df_axis = load_axis(condition)
        s = {"stacked": stacked, "overlay": overlay,
             "sigma_curve_position": sigma_curve_position(df_axis),
             "sigma_curve_per_axis": sigma_curve_position_per_axis(df_axis),
             "sigma_hop200": sigma_axis_position(df_axis, HOP)}
        s["sigma_hop200"]["mean_sigma"] = s["sigma_hop200"][AXES].mean(axis=1)
        s["cloud"] = df_axis[df_axis["config"] == CONFIG].copy()
        summaries[condition] = s
        del df_axis
        gc.collect()
        print(f"=== loaded {condition} (figures) ===")

    # stacked per-layer cumulative drift figures
    for layer, (measures, bounded) in STACKED_LAYER_SPECS.items():
        if type(bounded) == bool:
            bounded = [bounded] * len(measures)
        fig, axs = F.make_grid(
            width_mm=193, panels=(len(measures), 3), panel_aspect=0.5,
            margins=(0.10, 0.055, 0.985, 0.94), gutter=(0.05, 0.10),
            constrained=False, flatten=False, sharey="row", sharex=True,
        )
        condition_data = {c: summaries[c]["stacked"] for c in STACKED_CONDITIONS}
        for r, measure in enumerate(measures):
            plot_measure_row(
                axs[r], condition_data, STACKED_CONDITIONS, measure,
                bounded=bounded[r], show_title=(r == 0),
                show_xlabel=(r == len(measures) - 1),
            )
            axs[r, 0].set_ylabel(TITLES[measure], rotation=90)
        F.hide_interior_labels(axs, keep_bottom=True, keep_left=True,
                               apply_x=True, apply_y=True)
        F.save(fig, STACKED_PUB / f"fig_pilot_cum_{layer}",
               formats=("pdf", "png"), dpi_png=600, tight=True)
        F.file_dimensions(STACKED_PUB, f"fig_pilot_cum_{layer}", print_only=True)
        plt.close(fig)

    # headline figure (2x2), three conditions
    def pick_widening_seed(axis):
        d = summaries["direct"]["sigma_hop200"]
        w = summaries["vs_weighted"]["sigma_hop200"]
        d = d[d["config"] == CONFIG].set_index("msg_id")[axis]
        w = w[w["config"] == CONFIG].set_index("msg_id")[axis]
        diff = (w - d).sort_values(ascending=False)
        return diff.index[0]

    seed = pick_widening_seed("factual_to_narrative")
    print("widening seed:", seed)
    configs = sorted(summaries["direct"]["sigma_hop200"]["config"].unique())

    fig, axs = F.make_grid(
        width_mm=184, panels=(2, 2), panel_aspect=0.62,
        margins=(0.07, 0.06, 0.99, 0.955), gutter=(0.05, 0.42),
        constrained=False, flatten=True,
    )
    ylim = (-1.0, 1.0)
    plot_cloud(axs[0], summaries["direct"]["cloud"], seed,
               "factual_to_narrative", CONDITION_COLORS["direct"],
               CONDITION_LABELS["direct"], yaxis=True, ylim=ylim)
    plot_cloud(axs[1], summaries["vs_weighted"]["cloud"], seed,
               "factual_to_narrative", CONDITION_COLORS["vs_weighted"],
               CONDITION_LABELS["vs_weighted"], yaxis=False, ylim=ylim)
    for condition in CONDITIONS:
        c = summaries[condition]["sigma_curve_position"]
        axs[2].plot(c["hop"], c["mean_sigma"],
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition])
    axs[2].set_xlabel("Hop")
    axs[2].set_title(r"Mean between-chain variance $\Sigma$ over hops")
    axs[2].set_ylabel(r"Mean $\Sigma$")
    axs[2].legend(frameon=False)
    axs[2].set_ylim(-0.01, 0.1)
    x = np.arange(len(configs))
    for condition in CONDITIONS:
        s = summaries[condition]["sigma_hop200"]
        means = s.groupby("config")["mean_sigma"].mean().reindex(configs).to_numpy()
        axs[3].plot(x, means, color=CONDITION_COLORS[condition], marker="o",
                    ms=3, lw=1.2, label=CONDITION_LABELS[condition])
        for i, c in enumerate(configs):
            vals = s.loc[s["config"] == c, "mean_sigma"].to_numpy()
            axs[3].scatter(
                np.full_like(vals, x[i], dtype=float)
                + np.random.default_rng(0).uniform(-0.12, 0.12, len(vals)),
                vals, s=4, color=CONDITION_COLORS[condition], alpha=0.35,
                edgecolors="none")
    axs[3].set_xticks(
        x, [c.replace("temp", "T").replace("_topp", ", p=") for c in configs],
        rotation=45, ha="right")
    axs[3].set_title(rf"Mean $\Sigma$ at hop {HOP}")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.05), ncol=len(CONDITIONS),
               frameon=False, fontsize=7)
    F.save(fig, STACKED_PUB / "fig_sigma_headline", formats=("pdf", "png"),
           dpi_png=600, tight=True)
    F.file_dimensions(STACKED_PUB, "fig_sigma_headline", print_only=True)
    plt.close(fig)

    # overlay figures
    def _overlay_figure(metrics, titles_, fname, ylabel):
        fig, axs = F.make_grid(
            width_mm=186, panels=(1, len(metrics)), panel_aspect=0.62,
            constrained=False, flatten=True,
            margins=(0.07, 0.06, 0.99, 0.955), gutter=(0.05, 0.42),
            sharey=True,
        )
        for i, (ax, m) in enumerate(zip(axs, metrics)):
            for condition in CONDITIONS:
                c = summaries[condition]["overlay"]
                sub = c[c["metric"] == m].sort_values("hop")
                anchor = ANCHOR_VALUES.get(m)
                xs = ([0] + sub["hop"].tolist()) if anchor is not None else sub["hop"].tolist()
                ys = ([anchor] + sub["value"].tolist()) if anchor is not None else sub["value"].tolist()
                ax.plot(xs, ys, color=CONDITION_COLORS[condition], lw=1.2,
                        label=CONDITION_LABELS[condition])
            ax.set_xlabel("Hop")
            if i == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(titles_[m])
            ax.set_ylim(0, 1)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.05), ncol=len(CONDITIONS),
                   frameon=False, fontsize=7)
        F.save(fig, STACKED_PUB / fname, formats=("pdf", "png"),
               dpi_png=600, tight=True)
        F.file_dimensions(STACKED_PUB, fname, print_only=True)
        plt.close(fig)

    _overlay_figure(OVERLAY_METRICS, OVERLAY_TITLES, "fig_info_core", "Similarity")
    _overlay_figure(ENTAILMENT_METRICS, ENTAILMENT_TITLES,
                    "fig_info_entailment", "Probability")

    # per-axis Sigma(h) curves
    fig, axs = F.make_grid(
        width_mm=185, panels=(1, len(AXES)), panel_aspect=1.0,
        margins=(0.07, 0.08, 0.99, 0.94), gutter=(0.05, 0.4),
        constrained=False, flatten=True,
    )
    for i, (ax, axis) in enumerate(zip(axs, AXES)):
        for condition in CONDITIONS:
            c = summaries[condition]["sigma_curve_per_axis"]
            ax.plot(c["hop"], c[axis], color=CONDITION_COLORS[condition],
                    lw=1.2, label=CONDITION_LABELS[condition])
        ax.set_xlabel("Hop")
        ax.set_title(AXIS_LABELS[axis], fontsize=7)
        ax.set_ylim(-0.005, 0.05)
        if i > 0:
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel(r"Per-axis $\Sigma$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.2), ncol=len(CONDITIONS),
               frameon=False, fontsize=7)
    F.save(fig, STACKED_PUB / "fig_sigma_per_axis_curves",
           formats=("pdf", "png"), dpi_png=600, tight=True)
    F.file_dimensions(STACKED_PUB, "fig_sigma_per_axis_curves", print_only=True)
    plt.close(fig)

    # decoding sensitivity (merged three-panel figure)
    for condition in CONDITIONS:
        df_axis = load_axis(condition)
        summaries[condition]["slopes"] = drift_slopes(df_axis)
        del df_axis
        gc.collect()
    config_labels = [c.replace("temp", "T").replace("_topp", ", p=")
                     for c in configs]
    fig, axs = F.make_grid(
        width_mm=186, panels=(1, 3), panel_aspect=0.62,
        margins=(0.09, 0.13, 0.98, 0.96), gutter=(0.15, 0.42),
        constrained=False, flatten=True,
    )
    panels = [
        (lambda cond: _mu_by_config(summaries[cond]["slopes"], configs),
         r"Mean absolute drift $|\mu|$"),
        (lambda cond: _lasthop_by_config(cond, "cosine", configs),
         "Cosine similarity to seed"),
        (lambda cond: _lasthop_by_config(cond, "entail_ab", configs),
         "Seed entails hop"),
    ]
    for ax, (getter, title) in zip(axs, panels):
        for condition in CONDITIONS:
            ax.plot(x, getter(condition), color=CONDITION_COLORS[condition],
                    marker="o", ms=3, lw=1.2,
                    label=CONDITION_LABELS[condition])
        ax.set_xticks(x, config_labels, rotation=45, ha="right")
        ax.set_title(title, fontsize=7)
    axs[0].set_ylabel("Mean at hop 200")
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.25), ncol=len(CONDITIONS),
               frameon=False, fontsize=7)
    F.save(fig, STACKED_PUB / "fig_decoding_sensitivity",
           formats=("pdf", "png"), dpi_png=600, tight=True)
    F.file_dimensions(STACKED_PUB, "fig_decoding_sensitivity", print_only=True)
    plt.close(fig)


# --- examples stage --------------------------------------------------------


def latex_escape(s):
    return (
        s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
        .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
        .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def load_chain_texts(condition, config, msg_id, chain):
    sub_path = CONDITION_CONFIG[condition]["sub_path"]
    if sub_path is None:
        log_dir = LOGS_DIR / "gpt-4.1-nano" / config
    else:
        log_dir = LOGS_DIR / "gpt-4.1-nano" / sub_path / config
    fpath = log_dir / f"{msg_id}_chain_{chain:03d}.json"
    with open(fpath, "r", encoding="utf-8") as fp:
        chain_log = json.load(fp)
    return chain_log[0]["text"], chain_log[200]["text"]


def unique_hop200_texts(condition, config, msg_id):
    return [
        load_chain_texts(condition, config, msg_id, chain)[1]
        for chain in range(1, 10)
    ]


def _minmax(series):
    return {"min": float(series.min()), "max": float(series.max())}


def run_examples():
    import nltk

    with open(MSG_FILE, "r") as fp:
        seed_sentences = json.load(fp)
    seed_len = {
        "MSG_" + str(i + 1).zfill(3): x["length_tk"]
        for i, x in enumerate(seed_sentences["labels"])
    }

    summaries = {}
    exploratory = {"sigma_by_config": {}, "sigma_ratio_by_config": {},
                   "delta_len_by_config": {}, "k_unique_by_config": {},
                   "length_by_seed_len": {}, "examples": {}}
    for condition in CONDITIONS:
        df_axis = load_axis(condition)
        sh = sigma_axis_position(df_axis, HOP)
        sh["mean_sigma"] = sh[AXES].mean(axis=1)
        summaries[condition] = {"sigma_hop200": sh}
        exploratory["sigma_by_config"][condition] = _minmax(
            sh.groupby("config")["mean_sigma"].mean())
        del df_axis
        gc.collect()
    ratio_by_config = (
        summaries["vs_weighted"]["sigma_hop200"].groupby("config")["mean_sigma"].mean()
        / summaries["direct"]["sigma_hop200"].groupby("config")["mean_sigma"].mean())
    exploratory["sigma_ratio_by_config"] = _minmax(ratio_by_config)

    for condition in CONDITIONS:
        lh = pd.read_csv(stats_dir(condition) / "metrics_last_hop_cumulative.csv")
        lh = lh[lh["hop"] == HOP]
        for metric, key in [("delta_len", "delta_len_by_config"),
                            ("k_unique", "k_unique_by_config")]:
            exploratory[key][condition] = _minmax(
                lh[lh["metric"] == metric].groupby("config")["value"].mean())
        del lh

    # length_by_seed_len (vs_weighted only, as in the notebook)
    df = pd.read_csv(stats_dir("vs_weighted") / "metrics_last_hop_cumulative.csv")
    df = df[(df["config"] == CONFIG) & (df["hop"] == HOP)
            & (df["metric"].isin(["delta_len", "ratio_len"]))]
    piv = df.pivot_table(index=["msg_id", "chain"], columns="metric",
                         values="value").reset_index()
    piv["seed_len"] = piv["msg_id"].map(seed_len)
    short = piv[piv["seed_len"] <= 15]
    long = piv[piv["seed_len"] > 40]
    exploratory["length_by_seed_len"] = {
        "direct": None,
        "vs_argmax": None,
        "vs_weighted": {
            "short": {"delta_mean": float(short["delta_len"].mean()),
                      "ratio_mean": float(short["ratio_len"].mean())},
            "long": {"delta_abs": float(-long["delta_len"].mean()),
                     "ratio_mean": float(long["ratio_len"].mean())},
        },
    }

    # 8.1 sigma divergence example
    sh = summaries["vs_weighted"]["sigma_hop200"]
    sh = sh[sh["config"] == CONFIG]
    sigma_seed = sh.loc[sh["mean_sigma"].idxmax(), "msg_id"]
    sigma_seed_text, _ = load_chain_texts("vs_weighted", CONFIG, sigma_seed, 1)
    examples = {
        "sigma_divergence": {
            "msg_id": sigma_seed,
            "seed": latex_escape(sigma_seed_text),
            "n_direct_unique": len(set(unique_hop200_texts("direct", CONFIG, sigma_seed))),
            "n_argmax_unique": len(set(unique_hop200_texts("vs_argmax", CONFIG, sigma_seed))),
            "n_weighted_unique": len(set(unique_hop200_texts("vs_weighted", CONFIG, sigma_seed))),
            "weighted_hop200": [latex_escape(t) for t in
                                unique_hop200_texts("vs_weighted", CONFIG, sigma_seed)],
        }
    }

    # 8.2 fidelity loss example
    lh = pd.read_csv(stats_dir("vs_weighted") / "metrics_last_hop_cumulative.csv")
    lh = lh[(lh["config"] == CONFIG) & (lh["metric"] == "cosine") & (lh["hop"] == HOP)]
    fid = lh.loc[lh["value"].idxmin()]
    fid_seed, fid_hop200 = load_chain_texts("vs_weighted", CONFIG,
                                            fid["msg_id"], fid["chain"])
    examples["fidelity_loss"] = {
        "msg_id": fid["msg_id"], "chain": int(fid["chain"]),
        "seed": latex_escape(fid_seed), "hop200": latex_escape(fid_hop200),
        "cosine": float(fid["value"]),
    }

    # 8.3 abstract drift example
    df_axis = load_axis("vs_weighted")
    slopes = drift_slopes(df_axis)
    del df_axis
    gc.collect()
    slopes = slopes[slopes["config"] == CONFIG]
    abs_row = slopes.loc[slopes["abstract_to_concrete"].idxmin()]
    abs_seed, abs_hop200 = load_chain_texts(
        "vs_weighted", CONFIG, abs_row["msg_id"], abs_row["chain"])
    examples["abstract_drift"] = {
        "msg_id": abs_row["msg_id"], "chain": int(abs_row["chain"]),
        "seed": latex_escape(abs_seed), "hop200": latex_escape(abs_hop200),
        "slope": float(abs_row["abstract_to_concrete"]),
    }

    # 8.4 length shrink example
    lh_a = pd.read_csv(stats_dir("vs_argmax") / "metrics_last_hop_cumulative.csv")
    lh_a = lh_a[(lh_a["config"] == CONFIG) & (lh_a["metric"] == "delta_len")
                & (lh_a["hop"] == HOP)]
    len_row = lh_a.loc[lh_a["value"].idxmin()]
    len_seed, len_hop200 = load_chain_texts("vs_argmax", CONFIG,
                                            len_row["msg_id"], len_row["chain"])
    examples["length_shrink"] = {
        "msg_id": len_row["msg_id"], "chain": int(len_row["chain"]),
        "seed": latex_escape(len_seed), "hop200": latex_escape(len_hop200),
        "seed_tokens": len(nltk.word_tokenize(len_seed)),
        "hop200_tokens": len(nltk.word_tokenize(len_hop200)),
        "delta": float(len_row["value"]),
    }
    exploratory["examples"] = examples

    # sigma/mu macro sections into the three metrics_pub.json files
    headline = pd.read_csv(STATS_BASE / "sigma_exploratory" / "sigma_headline.csv")
    headline = headline.rename(columns={"arm": "condition"})
    sensitivity = pd.read_csv(
        STATS_BASE / "sigma_exploratory"
        / "sigma_sensitivity_drop_poison_cells.csv")
    w = headline.loc[headline["condition"] == "vs_weighted", "mean_sigma"].values[0]
    d = headline.loc[headline["condition"] == "direct", "mean_sigma"].values[0]
    a = headline.loc[headline["condition"] == "vs_argmax", "mean_sigma"].values[0]

    sigma_section = {
        "headline": {
            "direct": d, "vs_weighted": w, "vs_argmax": a,
            "ratio_weighted_direct": w / d, "ratio_weighted_argmax": w / a,
        },
        "sensitivity": {
            "ratio_weighted_direct": float(sensitivity["w_over_d"].values[0]),
            "ratio_weighted_argmax": float(sensitivity["w_over_a"].values[0]),
        },
        "per_axis": {
            condition: {
                ax: float(summaries[condition]["sigma_hop200"]
                          .groupby("config")[ax].mean().mean())
                for ax in AXES
            }
            for condition in CONDITIONS
        },
    }

    # mu section (the notebook also writes per-condition mu; the manuscript
    # macros descend desc.mu.<condition>.<axis>, and a missing "mu" key
    # reproduces as nil in json-read -> elisp listp(nil) is t -> the
    # getter's "List index must be int" error, which aborted the export
    # on 2026-10-03).
    mu_section = {}
    for condition in CONDITIONS:
        df_axis = load_axis(condition)
        mu = drift_summary(df_axis)
        del df_axis
        gc.collect()
        mu_section[condition] = (
            {ax: float(mu[ax]) for ax in AXES}
            | {"mean_abs": float(mu["mean_abs"])})

    for condition in CONDITIONS:
        out_json = (PUB_BASE / CONDITION_CONFIG[condition]["analysis_name"]
                    / "metrics_pub.json")
        pub = json.load(open(out_json, encoding="utf-8"))
        pub["sigma"] = sigma_section
        pub["mu"] = mu_section
        pub["exploratory"] = exploratory
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(pub, f, ensure_ascii=False, indent=2)
        print("Updated", out_json)


def write_manifest(stages):
    man = json.load(open(MANIFEST, encoding="utf-8")) if MANIFEST.exists() else {}
    man[datetime.now(UTC).isoformat()] = {
        "builder": "scripts/make_paper_outputs.py",
        "stages": stages,
        "labels": CONDITION_LABELS,
        "colors": CONDITION_COLORS,
        "metrics_version": METRICS_VERSION,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print("Wrote", MANIFEST)


def main():
    ap = argparse.ArgumentParser(
        description="Staged publication-output builder for the VS paper.")
    ap.add_argument("--stage", choices=["tables", "figures", "examples", "all"],
                    default="all")
    args = ap.parse_args()
    stages = []
    if args.stage in ("tables", "all"):
        run_tables()
        stages.append("tables")
    if args.stage in ("figures", "all"):
        run_figures()
        stages.append("figures")
    if args.stage in ("examples", "all"):
        run_examples()
        stages.append("examples")
    write_manifest(stages)


if __name__ == "__main__":
    main()
