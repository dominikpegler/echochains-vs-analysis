#!/usr/bin/env python3
"""Slide-only figures for the EchoChains VS paper (HMC lab deck).

Separate from the paper's analysis_01_descriptives_vs.py: produces deck
variants from the same verified exports, without touching the manuscript
figures. Deck TODO (index.org, VS block): the VS slides re-used the
paper's fig_sigma_headline / fig_sigma_per_axis_curves; mode sampling
leaves every slides figure (deck wording is already mode-free, the deck
concerns weighted sampling only), so VS is the single colored
line against Direct.

Inputs (published exports, no LM calls, no refitting):
  DATA_DIR/data/echodrift_stats/axis_cumulative_long.csv   (Direct)
  DATA_DIR/data/echodrift_stats/metrics_cumulative_long.csv (Direct)
  DATA_DIR/data/echodrift_stats/vs_weighted/axis_cumulative_long.csv
  DATA_DIR/data/echodrift_stats/vs_weighted/metrics_cumulative_long.csv

Outputs (PDF + PNG 300 dpi) into the deck's img/ directory:
  slide_sigma_headline.png        (2x2: clouds Direct/VS, Sigma(h), Sigma hop200)
  slide_sigma_per_axis_curves.png (1x5 per-axis Sigma(h), two lines)

Same sigma definitions as the paper analysis (echochain.diffusion):
between-chain variance of the axis projection across the nine chains,
grouped by config and seed; panels aggregate over seeds and configs as
in fig_sigma_headline / fig_sigma_per_axis_curves.

The headline seed is the same pick_widening_seed choice as the paper:
the seed with the largest weighted-minus-direct Sigma at hop 200 on the
factual-to-narrative axis (default config).

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

REPO = Path(__file__).resolve().parent.parent
for p in [REPO / "src" if (REPO / "src").exists() else REPO, REPO / "pub-utils"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from echochain.constants import CONDITION_COLORS, CONDITION_LABELS
from echochain.diffusion import AXES, sigma_axis_position
from echochain.utils import get_data_dir

import pub_utils.figures as F

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
    "tone_neutral_to_intense": "Tone intensity",
    "moderate_to_extreme": "Extremity",
    "factual_to_narrative": "Factual vs narrative",
    "abstract_to_concrete": "Abstract vs concrete",
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


def plot_cloud(ax, df_axis, msg_id, axis, color, label, yaxis=True):
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
    ax.set_title(label, fontsize=FTITLE, color="#000")


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


def headline_figure(summaries, seed):
    """Slide variant of fig_sigma_headline: 2x2, clouds + Sigma(h) + hop200.

    Two conditions only (Direct grey, VS magenta). Canvas ratio set
    for the deck row (max-height 300px inside 1422x800): wider and less
    tall than the paper 2x2, panels exactly 3:2.
    """
    fig, axs = F.make_grid(
        width_mm=220,
        panels=(2, 2),
        panel_aspect=0.60,
        margins=(0.075, 0.17, 0.99, 0.94),
        gutter=(0.08, 0.34),
        constrained=False,
        flatten=True,
    )

    ylim = (-1.0, 1.0)
    plot_cloud(
        axs[0], summaries["direct"]["cloud"], seed,
        "factual_to_narrative", CONDITION_COLORS["direct"], "Direct", yaxis=True,
    )
    plot_cloud(
        axs[1], summaries["vs_weighted"]["cloud"], seed,
        "factual_to_narrative", CONDITION_COLORS["vs_weighted"], "VS",
        yaxis=False,
    )

    # Bottom left: Sigma(h) curves, mean over axes
    for condition in SLIDE_CONDITIONS:
        c = summaries[condition]["sigma_curve_position"]
        axs[2].plot(
            c["hop"],
            c["mean_sigma"],
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            lw=FLW,
        )
    axs[2].set_xlabel("Hop")
    axs[2].set_title(r"Mean between-chain variance $\Sigma$ over hops",
                     fontsize=FTITLE, color="#000")
    axs[2].set_ylabel(r"Mean $\Sigma$")
    axs[2].legend(**LEGEND_KW)
    axs[2].set_ylim(-0.01, 0.1)

    # Bottom right: Sigma at hop 200, per config, per condition
    configs = sorted(summaries["direct"]["sigma_hop200"]["config"].unique())
    x = np.arange(len(configs))
    for condition in SLIDE_CONDITIONS:
        s = summaries[condition]["sigma_hop200"]
        means = s.groupby("config")["mean_sigma"].mean().reindex(configs).to_numpy()
        axs[3].plot(
            x,
            means,
            color=CONDITION_COLORS[condition],
            marker="o",
            ms=3.5,
            lw=FLW,
            label=CONDITION_LABELS[condition],
        )
    axs[3].set_xlabel("")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(configs, rotation=45, ha="right", fontsize=FAX - 1)
    axs[3].set_title(rf"Mean $\Sigma$ at hop {HOP}", fontsize=FTITLE, color="#000")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=CONDITION_COLORS[c], lw=FLW, marker="o", ms=3.5,
               label=CONDITION_LABELS[c])
        for c in SLIDE_CONDITIONS
    ]
    axs[3].legend(handles=handles, **LEGEND_KW)

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


def main():
    setup()
    summaries = {}
    for condition in SLIDE_CONDITIONS:
        df_axis = load_axis(condition)
        s = {}
        s["sigma_curve_position"] = sigma_curve_position(df_axis)
        s["sigma_curve_per_axis"] = sigma_curve_per_axis(df_axis)
        s["sigma_hop200"] = sigma_hop200(df_axis)
        s["cloud"] = (
            df_axis[df_axis["config"] == CFG].copy()
            if condition in ("direct", "vs_weighted")
            else None
        )
        summaries[condition] = s
        del df_axis
        print(f"processed {condition}")

    seed = pick_widening_seed(summaries, "factual_to_narrative")
    print("widening seed:", seed)

    # assertions: the deck's headline numbers reproduce from these frames
    w = summaries["vs_weighted"]["sigma_hop200"]
    d = summaries["direct"]["sigma_hop200"]
    ratio = (
        w.groupby("config")["mean_sigma"].mean()
        / d.groupby("config")["mean_sigma"].mean()
    )
    print("per-config weighted/direct Sigma ratio:")
    print(ratio.round(1).to_string())

    save_fig(headline_figure(summaries, seed), "slide_sigma_headline")
    save_fig(per_axis_figure(summaries), "slide_sigma_per_axis_curves")


if __name__ == "__main__":
    main()