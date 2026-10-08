#!/usr/bin/env python3
"""Slide-only figures for the EchoChains VS paper (HMC lab deck).

Separate from the paper's analysis_01_descriptives_vs.py: produces deck
variants from the same verified exports, without touching the manuscript
figures. Deck TODO (index.org, VS block): the VS slides re-used the
paper's fig_sigma_headline / fig_sigma_per_axis_curves; mode sampling
leaves every slides figure (deck wording is already mode-free, the deck
concerns weighted sampling only), so VS is the single colored
line against Direct.

Headline figure, 2026-10-03 revision (deck TODO resolved): the two
trajectory clouds share ONE panel on a shared y-axis (user decision: the
former side-by-side panels implied an identical scale without showing
the right panel's ticks), Direct grey vs VS magenta, and the legend
states each condition's Sigma value at hop 200 (seed/config/axis of the
plotted cloud; user note: "epsilon" meant Sigma).

Inputs (published exports, no LM calls, no refitting):
  DATA_DIR/data/echodrift_stats/axis_cumulative_long.csv   (Direct)
  DATA_DIR/data/echodrift_stats/metrics_cumulative_long.csv (Direct)
  DATA_DIR/data/echodrift_stats/vs_weighted/axis_cumulative_long.csv
  DATA_DIR/data/echodrift_stats/vs_weighted/metrics_cumulative_long.csv

Outputs (PDF + PNG 300 dpi) into the deck's img/ directory:
  slide_sigma_headline.png        (merged cloud + Sigma(h) + Sigma hop200;
                                   kept as the generator's 2x2 variant)
  slide_sigma_per_axis_curves.png (1x5 per-axis Sigma(h), two lines)
  slide_sigma_overview.png        (pass 5 headline: 1x6 -- five
                                   per-axis Sigma(h) + mean Sigma as
                                   the right-most panel; the pass-4
                                   trajectory row is removed, those
                                   curves live on the trajectories
                                   child)
  slide_sigma_cloud.png           (the merged cloud alone, sample named)
  slide_sigma_config.png          (Sigma at hop 200 per config + jitter)
  slide_drift_trajectories.png    (mean axis trajectory per condition)
  slide_drift_direction.png       (endpoint drift per axis, Direct vs VS,
                                   seed-clustered CIs since pass 4)

Same sigma definitions as the paper analysis (echochain.diffusion):
between-chain variance of the axis projection across the nine chains,
grouped by config and seed; panels aggregate over seeds and configs as
in fig_sigma_headline / fig_sigma_per_axis_curves.

The headline seed is the same pick_widening_seed choice as the paper:
the seed with the largest weighted-minus-direct Sigma at hop 200 on the
factual-to-narrative axis (default config). For MSG_024 that contrast
is extreme by construction: all nine Direct chains end at the seed
verbatim, so the cloud's Direct Sigma at hop 200 is near zero
(0.00011) while VS is 0.0728 (the paper's sigma-divergence example;
echochains_vs_paper.org). The legend therefore shows values that
belong to the plotted seed and config, not the deck-wide aggregates
(those are the bottom panels' subject).

Run (echochain conda env):
  conda run -n echochain python scripts/make_slide_figures.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
for p in [REPO / "src" if (REPO / "src").exists() else REPO, REPO / "pub-utils"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from echochain.constants import CONDITION_COLORS, CONDITION_LABELS
from echochain.diffusion import AXES, sigma_axis_position
from echochain.utils import get_data_dir

import pub_utils.figures as F
from pub_utils.figures import mm_to_in

# Slide-grade fonts (echochains/trustbandits slide-script pattern; the
# paper style uses base_font=7, projection needs more).
FAX = 9.0
FTITLE = 10.0
FLEG = 9.0
FLW = 1.1

LEGEND_KW = dict(
    frameon=False,
    fontsize=FLEG,
    columnspacing=1.6,
    handlelength=2.2,
    borderaxespad=0.0,
)

DATA = Path(get_data_dir())
STATS = DATA / "data/echodrift_stats"
DECK_IMG = Path("/home/user/write/postdoc-prep/hmc-lab/slides/img")
OUT = DATA / "pub/slides_vs"

CFG = "temp0.8_topp0.5"
HOP = 200

AXIS_LABELS = {
    "valence_neg_to_pos": "Valence",
    "tone_neutral_to_intense": "Intensity",
    "moderate_to_extreme": "Extremity",
    "factual_to_narrative": "Factual→narrative",
    "abstract_to_concrete": "Abstract→concrete",
}

# Conditions for the slides: Direct and VS only.
SLIDE_CONDITIONS = ["direct", "vs_weighted"]


def stats_dir(condition):
    return STATS / condition if condition != "direct" else STATS


def load_axis(condition):
    return pd.read_csv(stats_dir(condition) / "axis_cumulative_long.csv")


def sigma_curve_position(df_axis, axes=AXES):
    """Mean between-chain Sigma per hop, averaged over the given axes.

    Paper definition (sigma_curve_position): variance across chains
    within config and seed, averaged over seeds and configs.
    """
    var = df_axis.groupby(["config", "msg_id", "hop"])[axes].var(ddof=1)
    var["mean_sigma"] = var[axes].mean(axis=1)
    return var.groupby("hop")["mean_sigma"].mean().reset_index()


def sigma_curve_per_axis(df_axis):
    var = df_axis.groupby(["config", "msg_id", "hop"])[AXES].var(ddof=1)
    return var.groupby("hop")[AXES].mean().reset_index()


def sigma_hop200(df_axis, hop=HOP):
    s = sigma_axis_position(df_axis, hop)
    s["mean_sigma"] = s[AXES].mean(axis=1)
    return s


def pick_widening_seed(summaries, axis, config=CFG):
    """Seed with the largest weighted-minus-direct Sigma at hop (paper rule)."""
    d = summaries["direct"]["sigma_hop200"]
    w = summaries["vs_weighted"]["sigma_hop200"]
    d = d[d["config"] == config].set_index("msg_id")[axis]
    w = w[w["config"] == config].set_index("msg_id")[axis]
    diff = (w - d).sort_values(ascending=False)
    return diff.index[0]


def plot_cloud(ax, df_axis_by_condition, msg_id, axis, yaxis=True, legend=True,
               title=None):
    """One merged cloud: all chains of every SLIDE_CONDITIONS condition.

    2026-10-03: the former two side-by-side cloud panels (Direct left,
    VS right, y-ticks on the left panel only) implied a shared scale
    without showing it; the deck TODO asked for one plot on one scale
    and for the Sigma value in the legend. df_axis_by_condition maps
    each condition to its own axis frame (the Direct and VS exports
    have no condition column, so conditions are concatenated only for
    plotting order, not for the variance computation). Returns
    {condition: Sigma at hop 200} for the plotted seed/config/axis so
    the caller can put those values into the legend labels.
    """
    sigmas = {}
    for condition in SLIDE_CONDITIONS:
        sub = df_axis_by_condition[condition]
        sub = sub[(sub["msg_id"] == msg_id) & (sub["config"] == CFG)]
        for chain, g in sub.groupby("chain"):
            g = g.sort_values("hop")
            ax.plot(g["hop"], g[axis], color=CONDITION_COLORS[condition],
                    lw=0.6, alpha=0.5)
        sigmas[condition] = float(
            sub[sub["hop"] == HOP][axis].var(ddof=1)
        )
    ax.set_xlabel("Hop")
    ax.set_ylabel(AXIS_LABELS[axis] if yaxis else "")
    if title is not None:
        ax.set_title(title, fontsize=FTITLE, color="#000")
    if legend:
        handles = [
            Line2D(
                [0], [0],
                color=CONDITION_COLORS[condition], lw=FLW, alpha=0.9,
                label=f"{CONDITION_LABELS[condition]} "
                      rf"($\Sigma_{{200}}$ = {sigmas[condition]:.4f})",
            )
            for condition in SLIDE_CONDITIONS
        ]
        # Upper-left: above the Direct band (locked near 0.5) and clear
        # of the late-hop VS spike (upper right); lower-left collides
        # with the descending VS cloud. White backing: early VS lines
        # cross behind the text.
        ax.legend(
            handles=handles, loc="upper left", frameon=True,
            framealpha=0.9, facecolor="white", edgecolor="none",
            **{k: v for k, v in LEGEND_KW.items() if k != "frameon"},
        )
    return sigmas


def setup():
    F.setup_style(
        profile="nature",
        use_tex=False,
        base_font=FAX,
        minor_ticks=False,
        title_font_delta=0,
    )
    plt.rcParams.update(
        {
            "legend.fontsize": FLEG,
            "axes.labelsize": FAX,
            "xtick.labelsize": FAX,
            "ytick.labelsize": FAX,
        }
    )


def save_fig(fig, fname):
    OUT.mkdir(parents=True, exist_ok=True)
    F.save(fig, OUT / fname, formats=("pdf", "png"), dpi_png=300, tight=True)
    F.file_dimensions(OUT, fname, print_only=True)
    # deck copies
    for ext in ("png", "pdf"):
        src = OUT / f"{fname}.{ext}"
        dst = DECK_IMG / f"{fname}.{ext}"
        dst.write_bytes(src.read_bytes())
    print(f"deck copy: {DECK_IMG / (fname + '.png')}")
    plt.close(fig)


def cloud_figure(summaries, seed):
    """The merged trajectory cloud as its own figure (deck TODO pass 3).

    One seed, the default decoding config, nine chains per condition on
    the factual-vs-narrative axis; the sample is named in the title so
    the aggregation is explicit (user: the title must say what the
    trajectories are). The legend carries each condition's Sigma at hop
    200 (the seed-specific extreme values; 0.0001 vs 0.0728).
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(160), mm_to_in(88)))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.13)
    plot_cloud(
        ax, {c: summaries[c]["cloud"] for c in SLIDE_CONDITIONS}, seed,
        "factual_to_narrative", yaxis=True, title=None,
    )
    ax.set_title(
        rf"Seed {seed}, default decoding config, nine chains per condition",
        fontsize=FTITLE, color="#000",
    )
    return fig


def sigma_overview_figure(summaries):
    """All-axes headline: 1x6, five per-axis Sigma(h) + mean Sigma.

    Deck pass 4/5: the headline is ONE row of six Sigma panels -- the
    five per-axis Sigma(h) curves plus the mean-over-axes curve as the
    right-most panel (its y-band -0.01..0.1 differs from the per-axis
    -0.005..0.05 band, so it keeps its own y-ticks). The pass-4
    trajectory row was a misunderstanding of the user's request and is
    removed (those curves live on the "How the mean trajectory bends"
    child already).
    """
    fig, axs = F.make_grid(
        width_mm=250,
        panels=(1, 6),
        panel_aspect=0.80,
        margins=(0.10, 0.17, 0.995, 0.87),
        gutter=(0.10, 0.36),
        constrained=False,
        flatten=True,
    )

    # Panels 1-5: per-axis Sigma(h), averaged over seeds and configs
    for i, (ax, axis) in enumerate(zip(axs[:5], AXES)):
        for condition in SLIDE_CONDITIONS:
            c = summaries[condition]["sigma_curve_per_axis"]
            ax.plot(
                c["hop"],
                c[axis],
                color=CONDITION_COLORS[condition],
                lw=FLW,
                label=CONDITION_LABELS[condition],
            )
        ax.set_xlabel("Hop", fontsize=FAX - 1)
        ax.set_title(AXIS_LABELS[axis], fontsize=FTITLE, color="#000")
        ax.set_ylim(-0.005, 0.05)
        if i < 5:
            ax.tick_params(labelleft=False)
        if i == 4:
            ax.tick_params(labelsize=FAX - 1)
    axs[0].set_ylabel(r"Per-axis $\Sigma$")

    # Panel 6: mean Sigma over the five axes (own y-band, own y-ticks on
    # the RIGHT so they do not crowd the last per-axis panel's title)
    ax = axs[5]
    for condition in SLIDE_CONDITIONS:
        c = summaries[condition]["sigma_curve_position"]
        ax.plot(
            c["hop"],
            c["mean_sigma"],
            color=CONDITION_COLORS[condition],
            lw=FLW,
            label=CONDITION_LABELS[condition],
        )
    ax.set_xlabel("Hop", fontsize=FAX - 1)
    ax.set_title(r"Mean $\Sigma$", fontsize=FTITLE, color="#000")
    ax.set_ylim(-0.01, 0.1)
    ax.tick_params(labelleft=False, labelright=True, right=True, left=False)

    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], lw=FLW,
               label=CONDITION_LABELS[c])
        for c in SLIDE_CONDITIONS
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(SLIDE_CONDITIONS),
        **LEGEND_KW,
    )
    return fig


def headline_figure(summaries, seed, cloud_sigmas):
    """Slide variant of fig_sigma_headline, 2026-10-03 revision.

    Top: ONE merged trajectory cloud on a shared y-axis (Direct grey +
    VS magenta, nine chains each; legend carries each condition's Sigma
    at hop 200 for the plotted seed/config/axis). Bottom: Sigma(h)
    curves and Sigma at hop 200 per config, both aggregating over seeds
    and axes. Canvas ratio set for the deck row (max-height 300px
    inside 1422x800): wider and less tall than the paper 2x2.
    """
    fig = plt.figure(figsize=(mm_to_in(220), mm_to_in(94)))
    gs = fig.add_gridspec(
        2, 2,
        left=0.075, right=0.99, top=0.93, bottom=0.17,
        hspace=0.42, wspace=0.26,
    )
    ax_cloud = fig.add_subplot(gs[0, :])
    axs = [ax_cloud, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # Top: the merged cloud (both conditions in one panel, shared axis).
    # plot_cloud draws Direct first, then VS, from the two frames, and
    # recomputes the cloud Sigmas; the caller's dict must match
    # (asserted single source of truth).
    cloud_sigmas_actual = plot_cloud(
        axs[0], {c: summaries[c]["cloud"] for c in SLIDE_CONDITIONS}, seed,
        "factual_to_narrative", yaxis=True,
    )
    for condition in SLIDE_CONDITIONS:
        assert abs(cloud_sigmas_actual[condition] - cloud_sigmas[condition]) < 1e-9, (
            f"cloud Sigma mismatch for {condition}: "
            f"{cloud_sigmas_actual[condition]} vs {cloud_sigmas[condition]}"
        )

    # Bottom left: Sigma(h) curves, mean over axes
    for condition in SLIDE_CONDITIONS:
        c = summaries[condition]["sigma_curve_position"]
        axs[1].plot(
            c["hop"],
            c["mean_sigma"],
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            lw=FLW,
        )
    axs[1].set_xlabel("Hop")
    axs[1].set_title(r"Mean between-chain variance $\Sigma$ over hops",
                     fontsize=FTITLE, color="#000")
    axs[1].set_ylabel(r"Mean $\Sigma$")
    axs[1].legend(**LEGEND_KW)
    axs[1].set_ylim(-0.01, 0.1)

    # Bottom right: Sigma at hop 200, per config, per condition
    configs = sorted(summaries["direct"]["sigma_hop200"]["config"].unique())
    x = np.arange(len(configs))
    for condition in SLIDE_CONDITIONS:
        s = summaries[condition]["sigma_hop200"]
        means = s.groupby("config")["mean_sigma"].mean().reindex(configs).to_numpy()
        axs[2].plot(
            x,
            means,
            color=CONDITION_COLORS[condition],
            marker="o",
            ms=3.5,
            lw=FLW,
            label=CONDITION_LABELS[condition],
        )
    axs[2].set_xlabel("")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(configs, rotation=45, ha="right", fontsize=FAX - 1)
    axs[2].set_title(rf"Mean $\Sigma$ at hop {HOP}", fontsize=FTITLE, color="#000")

    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], lw=FLW, marker="o", ms=3.5,
               label=CONDITION_LABELS[c])
        for c in SLIDE_CONDITIONS
    ]
    axs[2].legend(handles=handles, **LEGEND_KW)

    return fig


def per_axis_figure(summaries):
    """Slide variant of fig_sigma_per_axis_curves: 1x5, two lines only."""
    fig, axs = F.make_grid(
        width_mm=230,
        panels=(1, len(AXES)),
        panel_aspect=0.78,
        margins=(0.07, 0.17, 0.99, 0.88),
        gutter=(0.06, 0.36),
        constrained=False,
        flatten=True,
    )
    for i, (ax, axis) in enumerate(zip(axs, AXES)):
        for condition in SLIDE_CONDITIONS:
            c = summaries[condition]["sigma_curve_per_axis"]
            ax.plot(
                c["hop"],
                c[axis],
                color=CONDITION_COLORS[condition],
                lw=FLW,
                label=CONDITION_LABELS[condition],
            )
        ax.set_xlabel("Hop")
        ax.set_title(AXIS_LABELS[axis], fontsize=FTITLE, color="#000")
        ax.set_ylim(-0.005, 0.05)
        if i > 0:
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel(r"Per-axis $\Sigma$")

    from matplotlib.lines import Line2D

    # Shared legend in the TOP band (the slide variant carries no suptitle,
    # so the top margin is free; a bottom legend collides with the large
    # slide-grade tick labels).
    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], lw=FLW,
               label=CONDITION_LABELS[c])
        for c in SLIDE_CONDITIONS
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(SLIDE_CONDITIONS),
        **LEGEND_KW,
    )
    return fig


def sigma_config_figure(summaries):
    """Backup child: mean Sigma at hop 200 per decoding config + jitter.

    Moved off the headline (pass 4, user: a detail someone might ask
    about). The jitter dots are restored to match the paper's panel
    (analysis_01_descriptives_vs.py: scatter of the 50 per-seed
    mean_sigma values per config and condition, rng(0), jitter window
    +-0.12, s=4, alpha 0.35); the slide variant had dropped them.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(200), mm_to_in(80)))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.24)
    configs = sorted(summaries["direct"]["sigma_hop200"]["config"].unique())
    x = np.arange(len(configs))
    for condition in SLIDE_CONDITIONS:
        s = summaries[condition]["sigma_hop200"]
        means = s.groupby("config")["mean_sigma"].mean().reindex(configs).to_numpy()
        ax.plot(
            x,
            means,
            color=CONDITION_COLORS[condition],
            marker="o",
            ms=3.5,
            lw=FLW,
            label=CONDITION_LABELS[condition],
        )
        # Paper-faithful jitter: one dot per seed (50) per config
        for i, cfg in enumerate(configs):
            vals = s.loc[s["config"] == cfg, "mean_sigma"].to_numpy()
            ax.scatter(
                np.full_like(vals, x[i], dtype=float)
                + np.random.default_rng(0).uniform(-0.12, 0.12, len(vals)),
                vals,
                s=14,
                color=CONDITION_COLORS[condition],
                alpha=0.35,
                edgecolors="none",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("temp", "T").replace("_topp", ", p=") for c in configs],
        rotation=45, ha="right", fontsize=FAX - 1,
    )
    ax.set_ylabel(rf"Mean $\Sigma$ at hop {HOP}")
    ax.legend(**LEGEND_KW)
    return fig


def drift_trajectories_figure(summaries, top_row=False):
    """Mean drift trajectories per axis (Direct vs VS), child figure.

    User point 2 (pass 4): the drift child mirror of the fidelity
    figure's mean-trajectory panels. One panel per axis; each line is
    the axis projection averaged over all seeds and configs; Direct
    grey, VS magenta. Paper 1's six-panel strips use shared y-limits;
    here each axis panel keeps -0.12..0.12 so the VS bends stay inside.
    """
    fig, axs = F.make_grid(
        width_mm=230,
        panels=(1, len(AXES)),
        panel_aspect=0.72,
        margins=(0.07, 0.17, 0.99, 0.88),
        gutter=(0.06, 0.36),
        constrained=False,
        flatten=True,
    )
    for i, (ax, axis) in enumerate(zip(axs, AXES)):
        for condition in SLIDE_CONDITIONS:
            c = summaries[condition]["drift_curves"]
            ax.plot(
                c.index,
                c[axis],
                color=CONDITION_COLORS[condition],
                lw=FLW,
                label=CONDITION_LABELS[condition],
            )
        ax.set_xlabel("Hop")
        ax.set_title(AXIS_LABELS[axis], fontsize=FTITLE, color="#000")
        ax.set_ylim(-0.12, 0.12)
        if i > 0:
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel("Mean axis score")

    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], lw=FLW,
               label=CONDITION_LABELS[c])
        for c in SLIDE_CONDITIONS
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(SLIDE_CONDITIONS),
        **LEGEND_KW,
    )
    return fig
    """Per-chain OLS slope of each axis on hop index (the paper's mu).

    Mirrors the paper analysis: slope per chain (config x seed x chain),
    in axis units per hop; the caller averages and bootstraps.
    Vectorized hop-by-hop within each chain via closed-form OLS.
    """
    out = []
    for _, g in df_axis.groupby(["config", "msg_id", "chain"]):
        g = g.sort_values("hop")
        h = g["hop"].to_numpy(dtype=float)
        h = h - h.mean()
        denom = float((h * h).sum())
        Y = g[list(axes)].to_numpy(dtype=float)
        Y = Y - Y.mean(axis=0)
        out.append((h[:, None] * Y).sum(axis=0) / denom)
    return pd.DataFrame(out, columns=list(axes))


def drift_direction_figure(direction):
    """Slide variant of fig_drift_direction: VS against Direct.

    Paper 1's direction child plots one dot + 95% CI per axis (mean
    signed drift, hop 200 minus hop 0). This variant keeps the same
    layout, x-label wording, and axis order, but groups the two
    conditions side by side per axis (Direct grey, VS magenta) so the
    direction change reads on one plot. 2026-10-03, deck TODO: where
    does VS drift on the semantic axes (Paper 2's real-drift case).
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(160), mm_to_in(88)))

    n_axes = len(AXES)
    y = np.arange(n_axes, dtype=float)
    offset = 0.19  # within-group offset, conditions above/below the tick
    for k, condition in enumerate(SLIDE_CONDITIONS):
        dd = direction[condition]
        means = np.array([dd["mean"][a] for a in AXES])
        lo = np.array([dd["ci95_lo"][a] for a in AXES])
        hi = np.array([dd["ci95_hi"][a] for a in AXES])
        yy = y + (offset if condition == "vs_weighted" else -offset)
        color = CONDITION_COLORS[condition]
        ax.hlines(yy, lo, hi, color=color, lw=1.6)
        ax.scatter(means, yy, s=24, color=color, zorder=3,
                   label=CONDITION_LABELS[condition])

    ax.axvline(0.0, color="#888888", lw=0.8, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([AXIS_LABELS[a] for a in AXES])
    ax.invert_yaxis()  # Paper 1's direction child lists Valence on top
    ax.set_xlabel("Mean signed drift, hop 200 minus hop 0 (axis units)")
    ax.legend(**LEGEND_KW)
    return fig


# Per-config robustness strips (2026-10-08, deck pass after Paper 1's):
# the same question Paper 1's config children answer, for the VS paper.
# X = the nine decoding configs; dot + 95% seed-clustered bootstrap CI
# per config and condition (unit of analysis = the seed, the per-seed
# mean over its nine chains within the config; bootstrap rng 2026,
# Paper 1's CI seed); both conditions per config (Direct grey at the
# left offset, VS magenta at the right offset). Canvas 12.3x3.3 in
# (ratio 3.73), matching Paper 1's two config strips. Inputs are the
# published exports (no LM calls, no refitting); the Direct side
# re-derives Paper 1's per-config numbers from the same CSVs, and
# _per_config_stats asserts the seed counts.

ROBUST_HOP = 200
ROBUST_BOOT = 2000
ROBUST_SEED = 2026
ROBUST_OFFSET = 0.15  # within-config offset, conditions left/right
ROBUST_CONFIGS = [
    f"temp{t}_topp{p}"
    for t in ("0.4", "0.8", "1.0")
    for p in ("0.3", "0.5", "0.8")
]
ROBUST_METRICS = ["cosine", "entail_consistent", "delta_len"]
ROBUST_PANEL = {
    "direct": ("#878D96", "Direct"),
    "vs_weighted": ("#EE5396", "VS"),
}


def _robust_frames(condition):
    """Per-seed frames for one condition: signed drift and last-hop metrics.

    Axis file: wide (config, msg_id) x axes with signed drift, hop 200
    minus hop 0, per seed = the mean over the seed's nine chains (same
    definition as Paper 1's drift_config_robustness, since mean(end) -
    mean(init) over equally many chains equals the mean per-chain
    difference). Metrics file: long, wide (config, msg_id) x
    metric with the per-seed mean at hop 200. No LM calls, no refitting.
    """
    ax = pd.read_csv(stats_dir(condition) / "axis_cumulative_long.csv")
    ax = ax[ax["hop"].isin([0, ROBUST_HOP])]
    wide = ax.pivot_table(
        index=["config", "msg_id"], columns="hop", values=AXES
    )
    drift = pd.DataFrame(
        {a: wide[(a, ROBUST_HOP)] - wide[(a, 0)] for a in AXES},
        index=wide.index,
    )
    met = pd.read_csv(stats_dir(condition) / "metrics_last_hop_cumulative.csv")
    met = met[
        (met["kind"] == "cumulative")
        & (met["hop"] == ROBUST_HOP)
        & met["metric"].isin(ROBUST_METRICS)
    ]
    metric = met.pivot_table(
        index=["config", "msg_id"], columns="metric", values="value"
    )
    return {"drift": drift, "metric": metric}


def _ci(vals):
    """Mean + 95% percentile bootstrap over seeds (rng = Paper 1's)."""
    vals = pd.Series(vals).dropna().to_numpy()
    n = len(vals)
    assert n == 50, f"expected 50 seeds, got {n}"
    rng = np.random.default_rng(ROBUST_SEED)
    draws = np.empty(ROBUST_BOOT)
    for b in range(ROBUST_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = vals[idx].mean()
    return vals.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def _per_config_stats(per_seed):
    """{ (metric, config) -> (mean, lo, hi) } from a per-seed frame.

    per_seed: (config, msg_id) index, one column per metric.
    """
    stats = {}
    configs = sorted(set(per_seed.index.get_level_values("config")))
    assert configs == ROBUST_CONFIGS, f"unexpected configs {configs}"
    for cfg in ROBUST_CONFIGS:
        sub = per_seed.xs(cfg, level="config")
        for metric in sub.columns:
            stats[(metric, cfg)] = _ci(sub[metric])
    return stats


def _config_strip(metrics, stats, ylabel_by_metric, ylim_by_metric,
                  zero_line=()):
    """1xN strip: x = the nine configs, two dots + 95% CIs per config.

    metrics: [(metric, panel title)]; stats: {condition -> {(metric,
    config) -> (mean, lo, hi)}} from _per_config_stats; zero_line:
    metrics whose panel draws the dashed zero reference.
    """
    n_cfg = len(ROBUST_CONFIGS)
    xlabels = [
        c.replace("temp", "T").replace("_topp", ", p=")
        for c in ROBUST_CONFIGS
    ]
    x = np.arange(n_cfg, dtype=float)
    fig, axs = plt.subplots(
        1, len(metrics), figsize=(12.3, 3.3), dpi=300
    )
    fig.subplots_adjust(left=0.04, right=0.995, top=0.965, bottom=0.30)
    for ax, (metric, title) in zip(axs, metrics):
        for condition, off in (("direct", -ROBUST_OFFSET),
                               ("vs_weighted", ROBUST_OFFSET)):
            color, name = ROBUST_PANEL[condition]
            means = np.array([
                stats[condition][(metric, c)][0] for c in ROBUST_CONFIGS
            ])
            los = np.array([
                stats[condition][(metric, c)][1] for c in ROBUST_CONFIGS
            ])
            his = np.array([
                stats[condition][(metric, c)][2] for c in ROBUST_CONFIGS
            ])
            if metric in zero_line:
                ax.axhline(0, color="#666666", lw=0.6, ls="--",
                           alpha=0.6, zorder=1)
            ax.errorbar(
                x + off,
                means,
                yerr=[means - los, his - means],
                fmt="o",
                ms=3.5,
                lw=1.4,
                capsize=2.0,
                color=color,
                ecolor=color,
                zorder=3,
                label=name,
            )
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.7)
        ax.tick_params(labelsize=10, width=0.7, colors="#000")
        ax.grid(linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.5, n_cfg - 0.5)
        lo, hi = ylim_by_metric[metric]
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10)
    handles = [
        Line2D([0], [0], color=ROBUST_PANEL[c][0], marker="o", lw=1.4,
               label=ROBUST_PANEL[c][1])
        for c in ("direct", "vs_weighted")
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        **LEGEND_KW,
    )
    axs[0].set_ylabel(ylabel_by_metric[metrics[0][0]], fontsize=10)
    fig.tight_layout(pad=0.4)
    return fig


def drift_config_vs_figure(axis_frames):
    """Backup child: per-axis endpoint drift per config, Direct vs VS.

    Paper 1's drift_config_robustness() layout (slide_drift_config.png),
    extended to both conditions: one panel per axis, x = the nine
    decoding configs, Direct grey / VS magenta, dot + 95% seed-clustered
    CI per config and condition (rng 2026, Paper 1's CI seed). Verified
    reads (2026-10-08): factual->narrative and abstract->concrete
    exclude zero in 9/9 configs under VS (means -0.078..-0.057 and
    -0.091..-0.059); the VS extremity reversal shows in every config's
    mean (-0.049..-0.019) but is CI-certain in only 1/9 (the qa note);
    valence and intensity cross zero everywhere, as in Paper 1.
    axis_frames: {} from _robust_frames()["drift"].
    """
    stats = {c: _per_config_stats(axis_frames[c]) for c in SLIDE_CONDITIONS}
    metrics = [(a, AXIS_LABELS[a]) for a in AXES]
    ylim = {a: (-0.13, 0.13) for a in AXES}
    return _config_strip(
        metrics, stats,
        {a: "mean signed drift (axis units)" for a in AXES},
        ylim, zero_line=AXES,
    )


def fidelity_config_vs_figure(metric_frames):
    """Backup child: seed fidelity and length per config, Direct vs VS.

    Paper 1's fidelity_config_robustness() layout (its 2026-10-08 1x3
    form), here with both conditions: cosine similarity and mutual
    entailment share a 0..1 y-axis (the VS collapse reads against
    Direct's near-seed level), the length panel keeps its own band and
    draws the zero line. Verified reads (2026-10-08): Direct cosine
    0.83..0.89 and entailment 0.62..0.82 per config; VS cosine
    0.57..0.61 and entailment 0.03..0.08 in every config; length
    contracts under VS (about -9.7 to -10.9 tokens, CIs exclude zero in
    9/9) while Direct grows (+2.7 to +8.8).
    metric_frames: {} from _robust_frames()["metric"].
    """
    stats = {c: _per_config_stats(metric_frames[c]) for c in SLIDE_CONDITIONS}
    metrics = [(m, t) for m, t in [
        ("cosine", "Cosine similarity"),
        ("entail_consistent", "Mutual entailment"),
        ("delta_len", "Length change (tokens)"),
    ]]
    ylim = {
        "cosine": (0.0, 1.0),
        "entail_consistent": (0.0, 1.0),
        "delta_len": (-14, 14),
    }
    return _config_strip(
        metrics, stats,
        {m: "mean at hop 200" for m, _ in metrics},
        ylim, zero_line=("delta_len",),
    )


def main():
    setup()
    summaries = {}
    for condition in SLIDE_CONDITIONS:
        df_axis = load_axis(condition)
        s = {}
        s["sigma_curve_position"] = sigma_curve_position(df_axis)
        s["sigma_curve_per_axis"] = sigma_curve_per_axis(df_axis)
        s["sigma_hop200"] = sigma_hop200(df_axis)
        s["cloud"] = df_axis[df_axis["config"] == CFG].copy()
        # Paper-1-identical direction stats (user point 1, pass 4): the
        # unit of analysis is the SEED, exactly as
        # echochains-sim-analysis seed_chain_analyses._seed_drift does:
        # per-seed mean endpoint minus mean initial state across that
        # seed's chains (all nine configs; 81 chains per seed), then a
        # 95% percentle bootstrap over the 50 seeds. The previous
        # chain-level bootstrap treated the 4050 chains as independent
        # and produced ~7x too-narrow CIs that disagreed with Paper 1's
        # figure for the same Direct data.
        sub = df_axis.sort_values("hop")
        ends = sub[sub["hop"] == HOP].groupby("msg_id")[AXES].mean()
        inits = sub[sub["hop"] == 0].groupby("msg_id")[AXES].mean()
        endpoint = (ends - inits).dropna()
        rng = np.random.default_rng(2026)  # Paper 1's CI seed
        boots = np.stack([
            endpoint.sample(len(endpoint), replace=True, random_state=i).mean()
            for i in range(2000)
        ])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        s["direction"] = {
            "mean": endpoint.mean(),
            "ci95_lo": pd.Series(lo, index=AXES),
            "ci95_hi": pd.Series(hi, index=AXES),
            "n_seeds": int(len(endpoint)),
        }
        # Mean drift trajectories (user point 2, pass 4): mean axis
        # projection per hop across all seeds and configs, per condition.
        s["drift_curves"] = df_axis.groupby("hop")[AXES].mean()
        summaries[condition] = s
        del df_axis
        print(f"processed {condition}")

    seed = pick_widening_seed(summaries, "factual_to_narrative")
    print("widening seed:", seed)

    # Sigmas for the merged cloud's legend: per condition at hop 200,
    # plotted seed/config/axis (MSG_024, default config, fact-narr).
    # plot_cloud recomputes them and the assert in headline_figure
    # guards the single source of truth.
    CLOUD_AXIS = "factual_to_narrative"
    cloud_sigmas = {
        condition: float(
            summaries[condition]["cloud"][
                (summaries[condition]["cloud"]["msg_id"] == seed)
                & (summaries[condition]["cloud"]["hop"] == HOP)
            ][CLOUD_AXIS].var(ddof=1)
        )
        for condition in SLIDE_CONDITIONS
    }
    print("cloud Sigma at hop 200 (seed/config/axis of the cloud):")
    for condition, v in cloud_sigmas.items():
        print(f"  {condition}: {v:.5f}")

    # assertions: the deck's headline numbers reproduce from these frames
    w = summaries["vs_weighted"]["sigma_hop200"]
    d = summaries["direct"]["sigma_hop200"]
    ratio = (
        w.groupby("config")["mean_sigma"].mean()
        / d.groupby("config")["mean_sigma"].mean()
    )
    print("per-config weighted/direct Sigma ratio:")
    print(ratio.round(1).to_string())

    save_fig(headline_figure(summaries, seed, cloud_sigmas), "slide_sigma_headline")
    save_fig(per_axis_figure(summaries), "slide_sigma_per_axis_curves")
    save_fig(sigma_overview_figure(summaries), "slide_sigma_overview")
    save_fig(cloud_figure(summaries, seed), "slide_sigma_cloud")
    save_fig(sigma_config_figure(summaries), "slide_sigma_config")
    save_fig(drift_trajectories_figure(summaries), "slide_drift_trajectories")

    # Direction figure: endpoint drift per condition, seed-clustered CIs
    direction = {c: summaries[c]["direction"] for c in SLIDE_CONDITIONS}
    print("endpoint drift, hop 200 minus hop 0 (axis units; Paper 1"
          " protocol: per-seed mean endpoint minus mean init over its"
          " 81 chains, CI = 95% bootstrap over the 50 seeds):")
    for condition in SLIDE_CONDITIONS:
        dd = direction[condition]
        for ax in AXES:
            print(f"  {condition:12s} {AXIS_LABELS[ax]:22s} "
                  f"{dd['mean'][ax]:+.4f}  [{dd['ci95_lo'][ax]:+.4f},"
                  f" {dd['ci95_hi'][ax]:+.4f}]")
    save_fig(drift_direction_figure(direction), "slide_drift_direction")

    # Per-config robustness strips (2026-10-08): one axis-level and one
    # metric-level strip, both conditions, seed-clustered CIs per
    # config. Inputs: the published hop 0/200 axis exports and the
    # last-hop metric exports (Direct = the root stats dir, VS =
    # vs_weighted/). The Direct side re-derives Paper 1's strips from
    # the same data; _robust_frames asserts the nine-config grid and
    # _ci asserts 50 seeds per cell.
    axis_frames = {c: _robust_frames(c)["drift"] for c in SLIDE_CONDITIONS}
    metric_frames = {c: _robust_frames(c)["metric"] for c in SLIDE_CONDITIONS}
    save_fig(drift_config_vs_figure(axis_frames), "slide_drift_config_vs",
             tight=True)
    save_fig(fidelity_config_vs_figure(metric_frames),
             "slide_fidelity_config_vs", tight=True)


if __name__ == "__main__":
    main()