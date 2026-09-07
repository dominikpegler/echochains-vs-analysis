# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py//py:percent
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.11
# ---

# %% [markdown]
# # EchoChain Drift Analysis (all conditions + Sigma)
#
# Single script covering the direct, vs_weighted, and vs_argmax conditions plus the
# mu/Sigma analysis. Replaces analysis_01_descriptives.py,
# analysis_01_descriptives_vs.py, and analysis_02_vs_sigma.py.
#
# Run once to regenerate all publication outputs:
#   python py/analysis_01_descriptives_vs.py
#
# Arms are processed sequentially to bound memory (each condition's long CSVs are
# ~4 GB in RAM); only small summaries are retained across conditions. The sigma/mu
# sections are written into all three metrics_pub.json files at the end, so
# there is no ordering dependency between descriptives and sigma outputs.

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import gc
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from echochain.diffusion import AXES, sigma_axis_position
from echochain.utils import get_data_dir
from echochain.icc_utils import compute_icc
from echochain.plot_utils import ci95
from echochain.constants import CONDITION_COLORS, CONDITION_LABELS

import publication_outputs.figures as F
from publication_outputs.tables import TableSpec, write_table

from transformers import logging

logging.set_verbosity_error()

WRITE_SHARDS = False

DATA_DIR = get_data_dir()
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")
STATS_BASE = Path(DATA_DIR, "data", "echodrift_stats")
PUB_BASE = Path(DATA_DIR, "pub")
METRICS_VERSION = "v1.0.0"
MSG_FILE = Path("seed_sentences_50.json")

HOP = 200
CONFIG = "temp0.8_topp0.5"
AXIS_LABELS = {
    "valence_neg_to_pos": "Valence",
    "tone_neutral_to_intense": "Tone intensity",
    "moderate_to_extreme": "Extremity",
    "factual_to_narrative": "Factual vs narrative",
    "abstract_to_concrete": "Abstract vs concrete",
}
CONDITIONS = ["direct", "vs_argmax", "vs_weighted"]

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

titles = {
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
    "delta_len",
    "ratio_len",
    "jaccard",
    "entity_f1",
    "triple_f1",
    "cosine",
    "entail_ab",
    "entail_ba",
    "entail_consistent",
    "k_unique",
]
OVERLAY_METRICS = ["cosine", "jaccard", "entail_ab", "entail_consistent"]
OVERLAY_TITLES = {
    "cosine": "Cosine similarity to seed",
    "jaccard": "Lexical Jaccard",
    "entail_ab": "NLI entailment (seed to hop)",
    "entail_consistent": "NLI entailment (consistent)",
}
ANCHOR_VALUES = {
    "delta_len": 0.0,
    "ratio_len": 1.0,
    "jaccard": 1.0,
    "entity_f1": 1.0,
    "triple_f1": 1.0,
    "cosine": 1.0,
    "entail_ab": 1.0,
    "entail_ba": 1.0,
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
    "delta_len",
    "ratio_len",
    "jaccard",
    "entity_f1",
    "triple_f1",
    "cosine",
    "entail_ab",
    "entail_ba",
    "entail_consistent",
]

# %% [markdown]
# ## 1. Load seed sentences

# %%
with open(MSG_FILE, "r") as fp:
    seed_sentences = json.load(fp)

seed_len = {
    "MSG_" + str(i + 1).zfill(3): x["length_tk"]
    for i, x in enumerate(seed_sentences["labels"])
}
seed_categories = {
    "MSG_" + str(i + 1).zfill(3): x for i, x in enumerate(seed_sentences["labels"])
}
seed_sentences_short = {
    "MSG_" + str(i + 1).zfill(3): x[:25] + "..."
    for i, x in enumerate(seed_sentences["sentences"])
}


def append_len(df):
    df_extra = df.loc[df["metric"] == "delta_len"].copy()
    df_extra["metric"] = "len"
    df_extra["value"] = df_extra["value"] + df_extra["msg_id"].map(seed_len)
    return pd.concat([df, df_extra], sort=False)


def add_category_cols(df):
    return df.merge(
        pd.DataFrame(seed_categories)
        .T.reset_index()
        .rename(columns={"index": "msg_id"}),
        how="left",
        on="msg_id",
    )


def add_seed_short(df):
    df["seed_short"] = df["msg_id"].map(seed_sentences_short)
    return df

# %% [markdown]
# ## 2. Per-condition processing
#
# Each condition is processed sequentially: load its long CSVs, write the last-hop
# tables, ICC table, and metrics_pub.json, stash small summaries for the
# cross-condition figures and the Sigma analysis, then free the big frames.

# %%
def stats_dir(condition):
    return STATS_BASE if condition == "direct" else STATS_BASE / condition


def load_metrics(condition):
    return pd.read_csv(stats_dir(condition) / "metrics_cumulative_long.csv")


def load_axis(condition):
    return pd.read_csv(stats_dir(condition) / "axis_cumulative_long.csv")


def last_hop_summary(df_one_cfg):
    idx = df_one_cfg.groupby(["msg_id", "chain", "metric"])["hop"].idxmax()
    d = df_one_cfg.loc[idx].copy()
    rows = []
    for m, g in d.groupby("metric"):
        mean, h95 = ci95(g["value"].values)
        rows.append(
            dict(metric=m, mean=mean, ci_lo=mean - h95, ci_hi=mean + h95, N=len(g))
        )
    out = pd.DataFrame(rows).set_index("metric").reindex(KEY_METRICS).reset_index()
    return out


def write_last_hop_table_for_config(
    df_cum, model, config, pub_dir, condition_label, condition
):
    tag = f"{model}_{config}"
    tag_safe = tag.replace(".", "p")
    fname = f"tab_lasthop_{tag_safe}"

    df_one = df_cum[(df_cum["model"] == model) & (df_cum["config"] == config)]
    tbl = last_hop_summary(df_one)
    disp = tbl[["metric", "mean", "ci_lo", "ci_hi", "N"]].copy()
    disp.columns = ["Metric", "Mean", "CI low", "CI high", "N"]
    disp["Metric"] = disp["Metric"].str.replace("_", " ")

    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} ({config.replace("_", " ")}) — {condition_label} — Last-hop cumulative""",
        label=f"tab:lasthop_{tag_safe}_{condition}",
        wrap="threeparttable",
        width="2col",
        use_tabularx=False,
        placement="htbp",
        use_siunitx=True,
        float_fmt="{:.3f}",
        use_index=False,
        fontsize_pt=(11, 13),
        header_overrides={
            "entail_ab": "entail seed→hop",
            "entail_ba": "entail hop→seed",
            "entail_consistent": "entail consistent",
        },
        numeric_as_r={"N"},
    )
    out_path = pub_dir / f"{fname}.tex"
    write_table(disp, out_path, spec)
    print("Wrote", out_path)
    return out_path


def write_combined_last_hop_table(
    df_cum, model, configs, pub_dir, condition_label, condition
):
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
        ["mean", "ci_lo", "ci_hi", "N"]
    ].reindex(KEY_METRICS)

    wide.index = wide.index.str.replace("_", " ")

    print(
        "Currently, we are only printing the mean column, N and CIs are dropped from the table.\n"
    )
    wide = wide.iloc[:, :9]  # mean only

    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} — {condition_label} — Last-hop cumulative across configs (temperature / top-p)""",
        label=f"tab:lasthop_{model.replace('.', 'p')}_all_{condition}",
        width="2col",
        wrap="threeparttable",
        use_tabularx=False,
        placement="tbp",
        use_siunitx=False,
        float_fmt="{:.3f}",
        fontsize_pt=(10, 12),
        use_index=False,
        group_cmidrules=True,
        group_cmidrules_skip={""},
        add_cmidrule=False,
        header_overrides={cfg: cfg.replace("_", r"\_") for cfg in configs},
    )

    out_path = pub_dir / f"{fname}.tex"
    write_table(wide.reset_index(), out_path, spec)
    print("Wrote", out_path)
    return out_path


def last_hop_series(df_one_cfg, metric):
    d = df_one_cfg[df_one_cfg["metric"] == metric]
    if d.empty:
        return pd.DataFrame(columns=["msg_id", "chain", "value"])
    idx = d.groupby(["msg_id", "chain"])["hop"].idxmax()
    return d.loc[idx, ["msg_id", "chain", "value"]].copy()


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
        tmp = s_cos.rename(columns={"outcome": "Cosine (logit)"})
        df_for_icc = tmp[["msg_id", "Cosine (logit)"]].copy()
        sigma2_between, sigma2_within, icc_val = compute_icc(
            df_for_icc, outcome="Cosine (logit)", group="msg_id"
        )
        rows.append(
            dict(
                config=config,
                outcome="Cosine (logit)",
                sigma2_between=float(sigma2_between),
                sigma2_within=float(sigma2_within),
                ICC=float(icc_val),
                N_groups=int(df_for_icc["msg_id"].nunique()),
                N_obs=int(df_for_icc.shape[0]),
            )
        )
    s_dl = last_hop_series(dcfg, "delta_len")
    if not s_dl.empty:
        tmp = s_dl.rename(columns={"value": r"$\Delta$ tokens"})
        df_for_icc = tmp[["msg_id", r"$\Delta$ tokens"]].copy()
        sigma2_between, sigma2_within, icc_val = compute_icc(
            df_for_icc, outcome=r"$\Delta$ tokens", group="msg_id"
        )
        rows.append(
            dict(
                config=config,
                outcome=r"$\Delta$ tokens",
                sigma2_between=float(sigma2_between),
                sigma2_within=float(sigma2_within),
                ICC=float(icc_val),
                N_groups=int(df_for_icc["msg_id"].nunique()),
                N_obs=int(df_for_icc.shape[0]),
            )
        )
    return rows


def build_icc_table(df_cum, model, configs):
    all_rows = []
    for cfg in configs:
        all_rows.extend(icc_for_config(df_cum, model, cfg))
    t = pd.DataFrame(all_rows)
    if not t.empty:
        t["config_order"] = t["config"].astype(str)
        t = t.sort_values(["config_order", "outcome"]).drop(columns=["config_order"])
    cols = [
        "config",
        "outcome",
        "sigma2_between",
        "sigma2_within",
        "ICC",
        "N_groups",
        "N_obs",
    ]
    return t[cols] if not t.empty else t


def write_icc_table(df_cum, model, configs, pub_dir, condition_label, condition):
    t = build_icc_table(df_cum, model, configs)
    if t.empty:
        raise RuntimeError(
            "ICC table is empty; check that df_cum has last-hop cosine and delta_len for the selected configs."
        )

    t = t.rename(
        columns={
            "config": "Config",
            "outcome": "Outcome",
            "sigma2_between": r"$\sigma^2_{\mathrm{between}}$",
            "sigma2_within": r"$\sigma^2_{\mathrm{within}}$",
            "ICC": "ICC",
            "N_groups": "$N_{groups}$",
            "N_obs": "$N$",
        }
    ).copy()
    t["Config"] = t["Config"].str.replace("_", " ")

    model_safe = model.replace(".", "p")
    fname = f"tab_icc_{model_safe}_configs"
    spec = TableSpec(
        caption=f"""{model.replace("_", " ")} — {condition_label} — Last-hop ICC by outcome and config (group = seed sentence). Outcomes computed at last hop per chain; ICC computed across messages (groups) using per-chain values.""",
        label=f"tab:icc_{model_safe}_configs_{condition}",
        wrap="threeparttable",
        width="2col",
        use_tabularx=False,
        placement="htbp",
        use_siunitx=True,
        fontsize_pt=(11, 13),
        float_fmt="{:.3f}",
        use_index=False,
        numeric_as_r={"$N_{groups}$", "$N$"},
    )
    out_path = pub_dir / f"{fname}.tex"
    write_table(t, out_path, spec)
    print("Wrote", out_path)
    return out_path


def metric_mean_ci_at_last(df, metric):
    idx = df[df["metric"] == metric].groupby(["msg_id", "chain"])["hop"].idxmax()
    vals = df.loc[idx, "value"].values
    mu, h = ci95(vals)
    return dict(
        mean=float(mu), ci_lo=float(mu - h), ci_hi=float(mu + h), N=int(len(vals))
    )


def sample_sizes(df):
    msgs = df["msg_id"].nunique()
    chains = df.groupby("msg_id")["chain"].nunique()
    hops = df["hop"].max()
    total_pairs = df[["msg_id", "chain", "hop"]].drop_duplicates().shape[0]
    return dict(
        n_messages=int(msgs),
        n_chains_mean=float(chains.mean()),
        n_chains_min=int(chains.min()),
        n_chains_max=int(chains.max()),
        hops_max=int(hops),
        total_pairs=int(total_pairs),
    )


def json_safe_key(cfg):
    return cfg.replace(".", "p")


def write_metrics_pub(df_cum, models, configs, pub_dir):
    pub = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics_version": METRICS_VERSION,
            "models": models,
            "configs": configs,
        },
        "config_key_map": {},
        "model_key_map": {},
        "per_model": {},
        "icc": {},
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
    out_json = pub_dir / "metrics_pub.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pub, f, ensure_ascii=False, indent=2)
    print("Wrote", out_json)


def sigma_curve_position(df_axis):
    var = df_axis.groupby(["config", "msg_id", "hop"])[AXES].var(ddof=1)
    var["composite"] = var[AXES].mean(axis=1)
    return var.groupby("hop")["composite"].mean().reset_index()


def sigma_curve_position_per_axis(df_axis):
    var = df_axis.groupby(["config", "msg_id", "hop"])[AXES].var(ddof=1)
    return var.groupby("hop")[AXES].mean().reset_index()


def drift_slopes(df_axis):
    rows = []
    for (config, msg_id, chain), grp in df_axis.groupby(["config", "msg_id", "chain"]):
        g = grp.sort_values("hop")
        h = g["hop"].to_numpy(dtype=float)
        row = {"config": config, "msg_id": msg_id, "chain": chain}
        for ax in AXES:
            y = g[ax].to_numpy(dtype=float)
            row[ax] = float(np.polyfit(h, y, 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def drift_summary(df_axis):
    slopes = drift_slopes(df_axis)
    mu = slopes[AXES].mean()
    mu["composite_abs"] = mu.abs().mean()
    return mu


def process_condition(condition):
    cfg = CONDITION_CONFIG[condition]
    sub_path = cfg["sub_path"]
    analysis_name = cfg["analysis_name"]
    condition_label = cfg["condition_label"]
    chosen_models = cfg["chosen_models"]
    chosen_configs = cfg["chosen_configs"]

    sdir = stats_dir(condition)
    sdir.mkdir(parents=True, exist_ok=True)
    shard_root = sdir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    pub_dir = PUB_BASE / analysis_name
    pub_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(DATA_DIR, "data", "echodrift_emb", analysis_name)

    if WRITE_SHARDS:
        if condition == "direct":
            from echochain.generate_stats import write_shards

            write_shards(
                logs_dir=LOGS_DIR,
                shard_root=shard_root,
                emb_dir=emb_dir,
                metrics_version=METRICS_VERSION,
            )
        else:
            from echochain.generate_stats_vs import write_shards

            write_shards(
                logs_dir=LOGS_DIR,
                shard_root=shard_root,
                sub_path=sub_path,
                emb_dir=emb_dir,
                metrics_version=METRICS_VERSION,
            )
    else:
        print(f"SKIP write_shards ({condition})")

    df_cum = load_metrics(condition)
    df_axis = load_axis(condition)

    df_cum = append_len(df_cum)
    df_cum = add_category_cols(df_cum)
    df_cum = add_seed_short(df_cum)

    models = (
        chosen_models
        if chosen_models is not None
        else sorted(df_cum["model"].unique().tolist())
    )
    configs = (
        chosen_configs
        if chosen_configs is not None
        else sorted(df_cum["config"].dropna().unique().tolist())
    )
    all_configs = sorted(df_cum["config"].dropna().unique().tolist())

    for model in models:
        for cfg in configs:
            print(model, cfg)
            write_last_hop_table_for_config(
                df_cum, model, cfg, pub_dir, condition_label, condition
            )
    for model in models:
        write_combined_last_hop_table(
            df_cum, model, all_configs, pub_dir, condition_label, condition
        )
    for model in models:
        write_icc_table(df_cum, model, configs, pub_dir, condition_label, condition)
    write_metrics_pub(df_cum, models, configs, pub_dir)

    s = {}
    stacked = df_cum[(df_cum["config"] == CONFIG) & (df_cum["model"].isin(models))]
    s["stacked"] = stacked[stacked["metric"].isin(STACKED_METRICS)][
        ["metric", "hop", "value"]
    ].copy()
    s["overlay"] = (
        stacked[stacked["metric"].isin(OVERLAY_METRICS)]
        .groupby(["metric", "hop"])["value"]
        .mean()
        .reset_index()
    )
    s["sigma_curve_position"] = sigma_curve_position(df_axis)
    s["sigma_curve_per_axis"] = sigma_curve_position_per_axis(df_axis)
    s["sigma_hop200"] = sigma_axis_position(df_axis, HOP)
    s["sigma_hop200"]["composite"] = s["sigma_hop200"][AXES].mean(axis=1)
    s["mu"] = drift_summary(df_axis)
    s["drift_slopes"] = drift_slopes(df_axis)
    s["cloud"] = (
        df_axis[df_axis["config"] == CONFIG].copy()
        if condition in ("direct", "vs_weighted")
        else None
    )

    def last_hop_mean_by_config(metric):
        d = df_cum[(df_cum["metric"] == metric) & (df_cum["hop"] == HOP)]
        return d.groupby("config")["value"].mean()

    s["last_hop_by_config"] = {
        "delta_len": last_hop_mean_by_config("delta_len"),
        "k_unique": last_hop_mean_by_config("k_unique"),
    }

    if condition == "vs_weighted":
        d = df_cum[
            (df_cum["config"] == CONFIG)
            & (df_cum["hop"] == HOP)
            & (df_cum["metric"].isin(["delta_len", "ratio_len"]))
        ]
        piv = d.pivot_table(
            index=["msg_id", "chain"], columns="metric", values="value"
        ).reset_index()
        piv["seed_len"] = piv["msg_id"].map(seed_len)
        short = piv[piv["seed_len"] <= 15]
        long = piv[piv["seed_len"] > 40]
        s["length_by_seed_len"] = {
            "short": {
                "delta_mean": float(short["delta_len"].mean()),
                "ratio_mean": float(short["ratio_len"].mean()),
            },
            "long": {
                "delta_abs": float(-long["delta_len"].mean()),
                "ratio_mean": float(long["ratio_len"].mean()),
            },
        }
    else:
        s["length_by_seed_len"] = None

    del df_cum, df_axis
    gc.collect()
    return s


summaries = {}
for condition in CONDITIONS:
    print(f"\n=== Processing condition: {condition} ===")
    summaries[condition] = process_condition(condition)

# %% [markdown]
# ## 3. Stacked per-layer cumulative drift figures
#
# One stacked figure per measurement layer (surface, factual, semantic) with
# rows for the three conditions: Direct (top), VS-weighted (middle),
# VS-argmax (bottom). Written into pub/descriptives_vs/ and referenced by
# the manuscript as fig:pilot-cum-{surface,factual,semantic}.

# %%
F.setup_style(
    profile="nature", use_tex=False, base_font=7, minor_ticks=False, title_font_delta=0
)
STACKED_CONDITIONS = ["direct", "vs_argmax", "vs_weighted"]
STACKED_PUB = PUB_BASE / "descriptives_vs"
STACKED_PUB.mkdir(parents=True, exist_ok=True)


def plot_measure_row(
    axs,
    condition_data,
    conditions,
    measure,
    *,
    bounded=False,
    show_title=True,
    show_xlabel=True,
):
    for ax, condition in zip(axs, conditions):
        df_layer = condition_data[condition]
        hops = sorted(df_layer["hop"].unique().tolist())
        xs, mu, lo, hi = [], [], [], []
        for h in hops:
            vals = df_layer[(df_layer["hop"] == h) & (df_layer["metric"] == measure)][
                "value"
            ].tolist()
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
        if measure == "delta_len":
            ax.axhline(0, color="#888", lw=0.6, ls="--", zorder=0)
        ax.tick_params(top=False, right=False)
    return axs


for layer, (measures, bounded) in STACKED_LAYER_SPECS.items():
    if type(bounded) == bool:
        bounded = [bounded] * len(measures)
    fig, axs = F.make_grid(
        width_mm=193,
        panels=(len(measures), 3),
        panel_aspect=0.5,
        margins=(0.10, 0.055, 0.985, 0.94),
        gutter=(0.05, 0.10),
        constrained=False,
        flatten=False,
        sharey="row",
        sharex=True,
    )
    condition_data = {c: summaries[c]["stacked"] for c in STACKED_CONDITIONS}
    for r, measure in enumerate(measures):
        show_xlabel = r == len(measures) - 1
        plot_measure_row(
            axs[r],
            condition_data,
            STACKED_CONDITIONS,
            measure,
            bounded=bounded[r],
            show_title=(r == 0),
            show_xlabel=show_xlabel,
        )
        axs[r, 0].set_ylabel(titles[measure], rotation=90)
    F.hide_interior_labels(
        axs, keep_bottom=True, keep_left=True, apply_x=True, apply_y=True
    )
    F.save(
        fig,
        STACKED_PUB / f"fig_pilot_cum_{layer}",
        formats=("pdf", "png"),
        dpi_png=600,
        tight=True,
    )
    F.file_dimensions(STACKED_PUB, f"fig_pilot_cum_{layer}", print_only=True)
    plt.show()
    plt.close()

# %% [markdown]
# ## 4. Sigma analysis: headline numbers

# %%
headline = pd.read_csv(STATS_BASE / "sigma_exploratory" / "sigma_headline.csv")
headline = headline.rename(columns={"arm": "condition"})
sensitivity = pd.read_csv(
    STATS_BASE / "sigma_exploratory" / "sigma_sensitivity_drop_poison_cells.csv"
)

w = headline.loc[headline["condition"] == "vs_weighted", "mean_composite_sigma"].values[
    0
]
d = headline.loc[headline["condition"] == "direct", "mean_composite_sigma"].values[0]
a = headline.loc[headline["condition"] == "vs_argmax", "mean_composite_sigma"].values[0]
print(f"headline: direct={d:.6f} weighted={w:.6f} argmax={a:.6f}")
print(f"weighted/direct ratio = {w/d:.2f}; weighted/argmax ratio = {w/a:.2f}")
print(
    f"sensitivity: w/d = {sensitivity['w_over_d'].values[0]:.2f}; w/a = {sensitivity['w_over_a'].values[0]:.2f}"
)

# %% [markdown]
# ## 5. Figures

# %%
F.setup_style(
    profile="nature", use_tex=False, base_font=7, minor_ticks=False, title_font_delta=0
)


def pick_widening_seed(axis):
    """Seed with the largest weighted-minus-direct between-chain variance at hop 200."""
    d = summaries["direct"]["sigma_hop200"]
    w = summaries["vs_weighted"]["sigma_hop200"]
    d = d[d["config"] == CONFIG].set_index("msg_id")[axis]
    w = w[w["config"] == CONFIG].set_index("msg_id")[axis]
    diff = (w - d).sort_values(ascending=False)
    return diff.index[0]


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


seed = pick_widening_seed("factual_to_narrative")
print("widening seed:", seed)

configs = sorted(summaries["direct"]["sigma_hop200"]["config"].unique())

# %% [markdown]
# ### 5.1. Composite between-chain diffusion figure
#
# Single 2x2 figure that combines the trajectory cloud (one seed, Direct vs
# VS-weighted), the Sigma(h) time course (composite over the five axes), and
# Sigma at hop 200 by decoding config. Panels: (a) Direct trajectories,
# (b) VS-weighted trajectories, (c) composite Sigma(h) curves, (d) composite
# Sigma at hop 200 by config.

# %%
fname = "fig_sigma_composite"

fig, axs = F.make_grid(
    width_mm=184,
    panels=(2, 2),
    panel_aspect=0.62,
    margins=(0.07, 0.06, 0.99, 0.955),
    gutter=(0.05, 0.42),
    constrained=False,
    flatten=True,
)

# Top row
ylim = (-1.0, 1.0)
plot_cloud(
    axs[0],
    summaries["direct"]["cloud"],
    seed,
    "factual_to_narrative",
    CONDITION_COLORS["direct"],
    "Direct",
    yaxis=True,
    ylim=ylim,
)
plot_cloud(
    axs[1],
    summaries["vs_weighted"]["cloud"],
    seed,
    "factual_to_narrative",
    CONDITION_COLORS["vs_weighted"],
    "VS-weighted",
    yaxis=False,
    ylim=ylim,
)

# Bottom left: Sigma(h) curves, composite only
ylim = (-0.01, 0.1)
for condition in CONDITIONS:
    c = summaries[condition]["sigma_curve_position"]
    axs[2].plot(
        c["hop"],
        c["composite"],
        color=CONDITION_COLORS[condition],
        label=CONDITION_LABELS[condition],
    )
axs[2].set_xlabel("Hop")
axs[2].set_title(r"Mean composite $\Sigma$ over hops")
axs[2].set_ylabel(r"Composite $\Sigma$")
axs[2].legend(frameon=False)
axs[2].set_ylim(ylim)

# Bottom right: Sigma at hop 200, per config, per condition
x = np.arange(len(configs))
for condition in CONDITIONS:
    s = summaries[condition]["sigma_hop200"]
    means = s.groupby("config")["composite"].mean().reindex(configs).to_numpy()
    axs[3].plot(
        x,
        means,
        color=CONDITION_COLORS[condition],
        marker="o",
        ms=3,
        lw=1.2,
        label=CONDITION_LABELS[condition],
    )
    for i, cfg in enumerate(configs):
        vals = s.loc[s["config"] == cfg, "composite"].to_numpy()
        axs[3].scatter(
            np.full_like(vals, x[i])
            + np.random.default_rng(0).uniform(-0.12, 0.12, len(vals)),
            vals,
            s=4,
            color=CONDITION_COLORS[condition],
            alpha=0.35,
            edgecolors="none",
        )
axs[3].set_xticks(
    x,
    [c.replace("temp", "T").replace("_topp", ", p=") for c in configs],
    rotation=45,
    ha="right",
)
axs[3].set_title(r"Composite $\Sigma$ at hop 200")
axs[3].set_yticklabels([])
axs[3].legend(frameon=False)
axs[3].set_ylim(ylim)

F.save(fig, STACKED_PUB / fname, formats=("pdf", "png"), dpi_png=600, tight=True)
F.file_dimensions(STACKED_PUB, fname, print_only=True)
plt.show()
plt.close()

# %% [markdown]
# ### 5.2. Information-preservation overlay (cosine, jaccard, entailment)

# %%
fname = "fig_info_preservation_overlay"
ncols = 2
fig, axs = F.make_grid(
    width_mm=186,
    panels=(2, ncols),
    panel_aspect=0.62,
    constrained=False,
    flatten=True,
    margins=(0.07, 0.06, 0.99, 0.955),
    gutter=(0.05, 0.42),
    sharey=True,
)
for i, (ax, m) in enumerate(zip(axs, OVERLAY_METRICS)):
    for condition in CONDITIONS:
        c = summaries[condition]["overlay"]
        sub = c[c["metric"] == m].sort_values("hop")
        anchor = ANCHOR_VALUES.get(m)
        xs = [0] + sub["hop"].tolist() if anchor is not None else sub["hop"].tolist()
        ys = (
            [anchor] + sub["value"].tolist()
            if anchor is not None
            else sub["value"].tolist()
        )
        ax.plot(
            xs,
            ys,
            color=CONDITION_COLORS[condition],
            lw=1.2,
            label=CONDITION_LABELS[condition],
        )
    ax.set_xlabel("Hop")
    if i % ncols == 0:
        ax.set_ylabel("Similarity")
    ax.set_title(OVERLAY_TITLES[m])
    ax.set_ylim(0, 1)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=len(CONDITIONS),
    frameon=False,
    fontsize=7,
)

F.save(fig, STACKED_PUB / fname, formats=("pdf", "png"), dpi_png=600, tight=True)
F.file_dimensions(STACKED_PUB, fname, print_only=True)
plt.show()
plt.close()

# %% [markdown]
# ### 5.3. Per-axis Sigma(h) curves (supplementary)
#
# One panel per semantic axis, three conditions overlaid, aggregated over all
# seeds and decoding configs. This is the aggregate view that complements the
# single-seed trajectory clouds in fig:sigma-composite; it shows that the
# between-chain diffusion increase under VS-weighted is not restricted to the
# factual-to-narrative axis.

# %%
fname = "fig_sigma_per_axis_curves"

fig, axs = F.make_grid(
    width_mm=185,
    panels=(1, len(AXES)),
    panel_aspect=1.0,
    margins=(0.07, 0.08, 0.99, 0.94),
    gutter=(0.05, 0.4),
    constrained=False,
    flatten=True,
)
for i, (ax, axis) in enumerate(zip(axs, AXES)):
    for condition in CONDITIONS:
        c = summaries[condition]["sigma_curve_per_axis"]
        ax.plot(
            c["hop"],
            c[axis],
            color=CONDITION_COLORS[condition],
            lw=1.2,
            label=CONDITION_LABELS[condition],
        )
    ax.set_xlabel("Hop")
    ax.set_title(AXIS_LABELS[axis], fontsize=7)
    ax.set_ylim(-0.005, 0.05)
    if i > 0:
        ax.tick_params(labelleft=False)

axs[0].set_ylabel(r"Per-axis $\Sigma$")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=len(CONDITIONS),
    frameon=False,
    fontsize=7,
)

F.save(fig, STACKED_PUB / fname, formats=("pdf", "png"), dpi_png=600, tight=True)
F.file_dimensions(STACKED_PUB, fname, print_only=True)
plt.show()
plt.close()

# %% [markdown]
# ## 6. Sigma tables

# %%
# 6.1. Headline Sigma table
headline_tbl = pd.DataFrame(
    {
        "Condition": [CONDITION_LABELS[a] for a in CONDITIONS],
        r"Composite $\Sigma$": [d, a, w],
        r"Ratio vs Direct": [1.0, a / d, w / d],
        r"Ratio vs VS-weighted": [d / w, a / w, 1.0],
    }
)
spec = TableSpec(
    caption="Composite between-chain variance $\\Sigma$ at hop 200 (mean across 9 decoding configs and 50 seeds), gpt-4.1-nano pilot. Composite $\\Sigma$ is the mean between-chain variance across the five semantic axes; ratios are computed on the composite.",
    label="tab:sigma_headline",
    wrap="threeparttable",
    width="2col",
    use_tabularx=False,
    placement="htbp",
    use_siunitx=True,
    fontsize_pt=(11, 13),
    float_fmt="{:.4f}",
    use_index=False,
)
write_table(headline_tbl, STACKED_PUB / "tab_sigma_headline.tex", spec)

# 6.2. Per-axis Sigma table
per_axis = pd.DataFrame({"Condition": [CONDITION_LABELS[a] for a in CONDITIONS]})
for ax in AXES:
    per_axis[AXIS_LABELS[ax]] = [
        summaries[a]["sigma_hop200"].groupby("config")[ax].mean().mean()
        for a in CONDITIONS
    ]
per_axis[r"Composite"] = [d, w, a]
spec = TableSpec(
    caption="Per-axis between-chain variance $\\Sigma$ at hop 200 (mean across configs and seeds), gpt-4.1-nano pilot.",
    label="tab:sigma_per_axis",
    wrap="threeparttable",
    width="2col",
    use_tabularx=False,
    placement="htbp",
    use_siunitx=True,
    fontsize_pt=(11, 13),
    float_fmt="{:.4f}",
    use_index=False,
)
write_table(per_axis, STACKED_PUB / "tab_sigma_per_axis.tex", spec)

# 6.3. Per-axis mu table
mu_tbl = pd.DataFrame({"Condition": [CONDITION_LABELS[a] for a in CONDITIONS]})
for ax in AXES:
    mu_tbl[AXIS_LABELS[ax]] = [summaries[a]["mu"][ax] for a in CONDITIONS]
mu_tbl[r"Composite $|\mu|$"] = [summaries[a]["mu"]["composite_abs"] for a in CONDITIONS]
spec = TableSpec(
    caption="Per-axis drift $\\mu$ (mean ordinary least squares slope of axis score on hop, per chain), gpt-4.1-nano pilot. Composite $|\\mu|$ is the mean absolute drift across the five axes.",
    label="tab:mu_per_axis",
    wrap="threeparttable",
    width="2col",
    use_tabularx=False,
    placement="htbp",
    use_siunitx=True,
    fontsize_pt=(11, 13),
    float_fmt="{:.5f}",
    use_index=False,
)
write_table(mu_tbl, STACKED_PUB / "tab_mu_per_axis.tex", spec)

# 6.4. Sensitivity table
sens_tbl = pd.DataFrame(
    {
        "Metric": ["Position"],
        r"$\Sigma$ Direct": [sensitivity["direct"].values[0]],
        r"$\Sigma$ VS-weighted": [sensitivity["vs_weighted"].values[0]],
        r"$\Sigma$ VS-argmax": [sensitivity["vs_argmax"].values[0]],
        r"Ratio weighted/direct": [sensitivity["w_over_d"].values[0]],
        r"Ratio weighted/argmax": [sensitivity["w_over_a"].values[0]],
    }
)
spec = TableSpec(
    caption="Sensitivity check: composite $\\Sigma$ at hop 200 after dropping all 18 poison-affected seed x config cells. 21 of 8,100 chains (0.26\\%) were excluded as deterministically crash-poisoned; the 18 affected seed x config cells are dropped entirely here.",
    label="tab:sigma_sensitivity",
    wrap="threeparttable",
    width="2col",
    use_tabularx=False,
    placement="htbp",
    use_siunitx=True,
    fontsize_pt=(11, 13),
    float_fmt="{:.4f}",
    use_index=False,
)
write_table(sens_tbl, STACKED_PUB / "tab_sigma_sensitivity.tex", spec)

print("wrote sigma tables to", STACKED_PUB)

# %% [markdown]
# ## 7. Extend metrics_pub.json with sigma and mu sections
#
# Writes the `sigma` and `mu` sections into all three condition-level
# `metrics_pub.json` files (descriptives, descriptives_vs, descriptives_vs_argmax)
# so that org macros referencing `desc_direct.sigma...` / `desc_argmax.sigma...`
# resolve. This is the last step, so there is no ordering dependency between
# the descriptives and the sigma/mu outputs.

# %%
sigma_section = {
    "headline": {
        "direct": d,
        "vs_weighted": w,
        "vs_argmax": a,
        "ratio_weighted_direct": w / d,
        "ratio_weighted_argmax": w / a,
    },
    "sensitivity": {
        "ratio_weighted_direct": float(sensitivity["w_over_d"].values[0]),
        "ratio_weighted_argmax": float(sensitivity["w_over_a"].values[0]),
    },
    "per_axis": {
        condition: {
            ax: float(
                summaries[condition]["sigma_hop200"].groupby("config")[ax].mean().mean()
            )
            for ax in AXES
        }
        for condition in CONDITIONS
    },
}
mu_section = {
    condition: {ax: float(summaries[condition]["mu"][ax]) for ax in AXES}
    | {"composite_abs": float(summaries[condition]["mu"]["composite_abs"])}
    for condition in CONDITIONS
}


def _minmax(series):
    return {"min": float(series.min()), "max": float(series.max())}


sigma_by_config = {
    condition: _minmax(
        summaries[condition]["sigma_hop200"].groupby("config")["composite"].mean()
    )
    for condition in CONDITIONS
}
ratio_by_config = (
    summaries["vs_weighted"]["sigma_hop200"].groupby("config")["composite"].mean()
    / summaries["direct"]["sigma_hop200"].groupby("config")["composite"].mean()
)

# %% [markdown]
# ## 8. Extreme examples for the manuscript
#
# Finds one illustrative extreme case per major result and stores the seed and
# final-hop texts in metrics_pub.json (exploratory.examples) so the manuscript
# can quote them via the {{{s(...)}}} macro.

# %%
def latex_escape(s):
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
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
    texts = []
    for chain in range(1, 10):
        _, h200 = load_chain_texts(condition, config, msg_id, chain)
        texts.append(h200)
    return texts


examples_section = {}

# 8.1. Sigma divergence: seed with the largest composite between-chain variance
# under VS-weighted at the default config.
sh = summaries["vs_weighted"]["sigma_hop200"]
sh = sh[sh["config"] == CONFIG]
sigma_seed = sh.loc[sh["composite"].idxmax(), "msg_id"]
sigma_seed_text, _ = load_chain_texts("vs_weighted", CONFIG, sigma_seed, 1)
examples_section["sigma_divergence"] = {
    "msg_id": sigma_seed,
    "seed": latex_escape(sigma_seed_text),
    "n_direct_unique": len(set(unique_hop200_texts("direct", CONFIG, sigma_seed))),
    "n_argmax_unique": len(set(unique_hop200_texts("vs_argmax", CONFIG, sigma_seed))),
    "n_weighted_unique": len(
        set(unique_hop200_texts("vs_weighted", CONFIG, sigma_seed))
    ),
    "weighted_hop200": [
        latex_escape(t) for t in unique_hop200_texts("vs_weighted", CONFIG, sigma_seed)
    ],
}

# 8.2. Fidelity loss: lowest last-hop cosine under VS-weighted at the default config.
lh = pd.read_csv(stats_dir("vs_weighted") / "metrics_last_hop_cumulative.csv")
lh = lh[(lh["config"] == CONFIG) & (lh["metric"] == "cosine")]
fid = lh.loc[lh["value"].idxmin()]
fid_seed, fid_hop200 = load_chain_texts(
    "vs_weighted", CONFIG, fid["msg_id"], fid["chain"]
)
examples_section["fidelity_loss"] = {
    "msg_id": fid["msg_id"],
    "chain": int(fid["chain"]),
    "seed": latex_escape(fid_seed),
    "hop200": latex_escape(fid_hop200),
    "cosine": float(fid["value"]),
}

# 8.3. Abstract drift: most negative abstract_to_concrete slope under VS-weighted.
slopes = summaries["vs_weighted"]["drift_slopes"]
slopes = slopes[slopes["config"] == CONFIG]
abs_row = slopes.loc[slopes["abstract_to_concrete"].idxmin()]
abs_seed, abs_hop200 = load_chain_texts(
    "vs_weighted", CONFIG, abs_row["msg_id"], abs_row["chain"]
)
examples_section["abstract_drift"] = {
    "msg_id": abs_row["msg_id"],
    "chain": int(abs_row["chain"]),
    "seed": latex_escape(abs_seed),
    "hop200": latex_escape(abs_hop200),
    "slope": float(abs_row["abstract_to_concrete"]),
}

# 8.4. Length shrink: largest last-hop token loss under VS-argmax at the default config.
lh_a = pd.read_csv(stats_dir("vs_argmax") / "metrics_last_hop_cumulative.csv")
lh_a = lh_a[(lh_a["config"] == CONFIG) & (lh_a["metric"] == "delta_len")]
len_row = lh_a.loc[lh_a["value"].idxmin()]
len_seed, len_hop200 = load_chain_texts(
    "vs_argmax", CONFIG, len_row["msg_id"], len_row["chain"]
)
examples_section["length_shrink"] = {
    "msg_id": len_row["msg_id"],
    "chain": int(len_row["chain"]),
    "seed": latex_escape(len_seed),
    "hop200": latex_escape(len_hop200),
    "seed_tokens": len(nltk.word_tokenize(len_seed)),
    "hop200_tokens": len(nltk.word_tokenize(len_hop200)),
    "delta": float(len_row["value"]),
}

exploratory_section = {
    "sigma_by_config": sigma_by_config,
    "sigma_ratio_by_config": _minmax(ratio_by_config),
    "delta_len_by_config": {
        condition: _minmax(summaries[condition]["last_hop_by_config"]["delta_len"])
        for condition in CONDITIONS
    },
    "k_unique_by_config": {
        condition: _minmax(summaries[condition]["last_hop_by_config"]["k_unique"])
        for condition in CONDITIONS
    },
    "length_by_seed_len": {
        condition: summaries[condition]["length_by_seed_len"]
        for condition in CONDITIONS
    },
    "examples": examples_section,
}

for condition in CONDITIONS:
    out_json = (
        PUB_BASE / CONDITION_CONFIG[condition]["analysis_name"] / "metrics_pub.json"
    )
    if not out_json.exists():
        print("Skipping missing", out_json)
        continue
    pub = json.load(open(out_json, encoding="utf-8"))
    pub["sigma"] = sigma_section
    pub["mu"] = mu_section
    pub["exploratory"] = exploratory_section
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pub, f, ensure_ascii=False, indent=2)
    print("Wrote", out_json)

# %%
