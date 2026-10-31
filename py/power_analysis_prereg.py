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
# # How much evidence can the confirmatory study produce?
#
# This notebook answers one question: if the confirmatory study runs at the
# planned size, how often do the preregistered decision rules find the
# pilot's effects, and how often do they raise a false alarm when no effect
# is there? The planned size is 90 seed texts, 5 chains per text, and 200
# transmitted hops per condition and task, for each model.
#
# The simulation works from the pilot's measured chains, not from invented
# distributions. For one simulated study it resamples seed texts from the
# pilot's 50 with repetition, and for each resampled seed it draws the
# study's 5 chains with repetition from that seed's 9 measured pilot
# chains. Resampling seeds stands in for the wider population of texts;
# drawing chains carries the measurement noise of a 5-chain spread, which a
# variance rule alone would get wrong where chain endings repeat or cluster.
# The arms of a simulated study use independent chain draws, as they do in
# the real study, and stay paired through the same seed texts.
#
# Three limits define what the answers mean. First, the pilot covers one
# small model and the paraphrase task; other models and tasks are covered
# by labeled scenario copies of the same pools, never by invented numbers.
# Second, the Multi-k uniform control has no pilot data at all; its
# assumptions are stated as labeled scenarios. Third, the preregistered
# estimators and decision rules are simulated as the spec table in the
# Analysis section defines them, so the reported chances apply to those
# rules as written, for a pilot-like population. Every pass rate comes from
# 2,000 simulated studies; at this count, chance moves the printed
# percentages by about one point.

# %%
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(
    os.environ.get("DATA_DIR",
                   str(Path.home() / "Dropbox/data_export/echochains")))
STATS = DATA_DIR / "data" / "echodrift_stats"
AXES = [
    "valence_neg_to_pos",
    "tone_neutral_to_intense",
    "moderate_to_extreme",
    "factual_to_narrative",
    "abstract_to_concrete",
]
IA, IF = 3, 4      # preregistered H2b axes: abstract-to-concrete,
                   # factual-to-narrative
CONFIG = "temp0.8_topp0.5"
BOUND = 0.005      # equivalence bound, axis units per hop
DESIGNS = [(50, 9), (75, 6), (90, 5), (112, 4), (112, 5), (150, 3)]
LAMBDAS = [0.0, 0.5, 1.0]   # effect strength: 0 none, 1 pilot-sized
B = 2000           # simulated studies per cell
SEED = 20261031
ALPHA = 0.05       # Benjamini-Hochberg false discovery rate

# %% [markdown]
# ## Pilot pools
#
# Per chain, the simulation uses four pilot quantities, all at the primary
# decoding setting (temperature 0.8, top-p 0.5): the axis score at hop 200,
# the OLS slope of each axis score on hop index, the last-hop cosine, and
# the two directional entailment scores. Direct and VS contribute their
# arms; the verbalized mode contributes nothing, because the confirmatory
# study does not collect it.

# %%
def load_axis(rel):
    cols = ["config", "msg_id", "chain", "hop"] + AXES
    return pd.read_csv(STATS / rel, usecols=cols)


def chain_slopes(df):
    """Per (config, seed, chain): OLS slope of each axis score on hop index.

    Centered least squares; each chain is fitted over the rows it has (a
    crash-truncated chain stops one hop early).
    """
    base = ["config", "msg_id", "chain"]
    gg = df.sort_values(base + ["hop"]).copy()
    gg["hc"] = gg["hop"] - gg.groupby(base)["hop"].transform("mean")
    s2 = gg.groupby(base)["hc"].agg(lambda x: (x ** 2).sum()).rename("hc2")
    sums = []
    for ax in AXES:
        c = gg[ax] - gg.groupby(base)[ax].transform("mean")
        s = (c * gg["hc"]).groupby([gg[b] for b in base]).sum().rename(ax)
        sums.append(s)
    out = pd.concat(sums, axis=1).join(s2)
    for ax in AXES:
        out[ax] = out[ax] / out["hc2"]
    return out.reset_index()[base + AXES]


def pool_hop200(df, cfg=None):
    """Chain axis scores at hop 200: (50 seeds, 9 chains, 5 axes).

    cfg=None keeps every config (a dict per config, see pools_per_config);
    a config name restricts the pool to that setting.
    """
    d = df.loc[df["hop"] == 200]
    if cfg is not None:
        d = d.loc[d["config"] == cfg]
    wide = d.pivot_table(index=["msg_id", "chain"], values=AXES)
    wide = wide.reorder_levels(["msg_id", "chain"]).sort_index()
    sh = wide.index.get_level_values("msg_id").nunique()
    assert wide.shape[0] == sh * 9, "incomplete (seed, chain) grid at hop 200"
    return wide.to_numpy().astype(float).reshape(sh, 9, len(AXES))


def pools_per_config(df):
    """One (50, 9, 5) pool per decoding config."""
    return {cfg: pool_hop200(df, cfg) for cfg in sorted(df["config"].unique())}


def pool_last_hop(metric):
    """Per-chain last-hop values for one metric at the fixed setting:
    (50 seeds, 9 chains), per arm."""
    out = {}
    for arm in ("direct", "vs_weighted"):
        rel = ("vs_weighted/metrics_last_hop_cumulative.csv"
               if arm == "vs_weighted" else "metrics_last_hop_cumulative.csv")
        df = pd.read_csv(STATS / rel, usecols=["config", "msg_id", "chain",
                                              "metric", "value"])
        d = df[(df["metric"] == metric) & (df["config"] == CONFIG)]
        wide = d.pivot_table(index=["msg_id", "chain"], values="value")
        wide = wide.reorder_levels(["msg_id", "chain"]).sort_index()
        sh = wide.index.get_level_values("msg_id").nunique()
        assert wide.shape[0] == sh * 9, "incomplete (seed, chain) grid"
        out[arm] = wide.to_numpy().astype(float).reshape(sh, 9)
    return out


axis_dir = load_axis("axis_cumulative_long.csv")
axis_vs = load_axis("vs_weighted/axis_cumulative_long.csv")
slopes_dir = chain_slopes(axis_dir)
slopes_vs = chain_slopes(axis_vs)
scores_dir = pool_hop200(axis_dir, CONFIG)
scores_vs = pool_hop200(axis_vs, CONFIG)
slopes_fixed = {
    "direct": chain_slopes(axis_dir.loc[axis_dir["config"] == CONFIG]),
    "vs": chain_slopes(axis_vs.loc[axis_vs["config"] == CONFIG]),
}
cos_pools = pool_last_hop("cosine")
ent_ab_pools = pool_last_hop("entail_ab")
ent_ba_pools = pool_last_hop("entail_ba")

# %% [markdown]
# The pools tie to the published pipeline values as follows. The fixed-
# setting spread sits at the Direct level under Direct and about ten times
# higher under VS; the grid average reproduces the paper's 8.52 ratio; the
# drift values reproduce the published per-setting drift; the similarity
# and entailment means reproduce the published last-hop tables.

# %%
def grid_sigma(df):
    s = df.loc[df["hop"] == 200]
    per_seed = s.groupby(["config", "msg_id"])[AXES].var(ddof=1)
    return float(per_seed.mean(axis=1).mean())


sig_d = scores_dir.var(axis=1, ddof=1).mean()
sig_v = scores_vs.var(axis=1, ddof=1).mean()
print(f"spread, fixed setting: Direct {sig_d:.5f}  VS {sig_v:.5f}  "
      f"ratio {sig_v / sig_d:.2f}")
print(f"spread, grid average:  Direct {grid_sigma(axis_dir):.5f}  "
      f"VS {grid_sigma(axis_vs):.5f}  "
      f"ratio {grid_sigma(axis_vs) / grid_sigma(axis_dir):.2f}")


def config_drift(slopes):
    """Headline drift per config: signed per-axis slopes averaged across
    chains first, absolute value after, mean across the axes."""
    return slopes.groupby("config")[AXES].mean().abs().mean(axis=1)


fdir = config_drift(slopes_dir)
fvs = config_drift(slopes_vs)
print(f"drift, fixed setting:  Direct {fdir[CONFIG]:.2e}  "
      f"VS {fvs[CONFIG]:.2e}")
print(f"drift, grid average:   Direct {fdir.mean():.2e}  VS {fvs.mean():.2e}")
print(f"last-hop cosine:       Direct {cos_pools['direct'].mean():.3f}  "
      f"VS {cos_pools['vs_weighted'].mean():.3f}")
print(f"entail ab / ba, Direct: {ent_ab_pools['direct'].mean():.3f} / "
      f"{ent_ba_pools['direct'].mean():.3f}")

# %% [markdown]
# ## Estimators and decision rules
#
# Every preregistered quantity is computed per seed; conditions are compared
# paired by seed, with a paired t interval over the seed-level differences.
# The five preregistered tests of a task:
#
# - **Spread (H1).** Per seed and axis, the variance of the axis scores
#   across the seed's chains, averaged over the five axes. Direction: VS
#   above Direct.
# - **Drift equivalence (H1).** The per-seed mean absolute drift, built
#   from the signed chain-average per-axis slopes. Rule: the 95% interval
#   of the VS-minus-Direct differences lies entirely inside $\pm 0.005$
#   axis units per hop. An interval criterion, outside the test family.
# - **Similarity (H2a).** Per seed, the mean last-hop cosine. Direction:
#   VS below Direct.
# - **Per-axis drift (H2b).** Per seed, the signed chain-average slope on
#   each preregistered axis. Direction: below 0 on both axes.
# - **Control contrast (H3).** The spread estimator against the Multi-k
#   uniform control. Direction: VS above control.
#
# A task's five tests enter the per-model Benjamini-Hochberg family over
# the four tasks (20 tests, false discovery rate 0.05). A test counts as
# found when the adjusted test rejects and the estimate points the
# preregistered way. The H2b asymmetry rule (seed-to-hop minus hop-to-seed
# entailment, VS minus Direct, unadjusted 95% interval above 0) and the
# equivalence rule are interval criteria, evaluated outside the family.

# %% [markdown]
# ## Scenario machinery
#
# The effect strength runs on one dial, $\lambda \in \{0, 0.5, 1\}$. At
# $\lambda = 0$ the VS arm reproduces the Direct pool seed by seed, so the
# whole 20-test family is null and the run measures false alarms. At
# $\lambda = 1$ the VS arm is the pilot's own measured values. At 0.5 the
# seed-level effect is halfway: synthetic VS chain values are the Direct
# pool plus half the seed-level VS-minus-Direct offset, and the spread
# scenario rescales the VS chains' within-seed deviations so each seed's
# variance interpolates between its Direct and pilot-VS values. Each seed
# keeps its own level throughout; nothing in the scenarios invents a
# distribution.
#
# The Multi-k uniform control has no pilot data. Two labeled scenarios
# stand in for it: the control behaves like Direct (power scenario for
# H3), or like VS (the scenario under which probability weighting adds
# nothing beyond the joint list, so H3 is null).

# %%
def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def paired_check(diff):
    """(p, mean, half-width) of the seed-level paired differences."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    se = diff.std(ddof=1) / math.sqrt(n)
    if se == 0.0:
        return 1.0, float(diff.mean()), 0.0
    tval = float(diff.mean() / se)
    p = 2 * stats.t.sf(abs(tval), n - 1)
    hw = float(stats.t.ppf(0.975, n - 1) * se)
    return p, float(diff.mean()), hw


def build_arms(lam):
    """Chain-level pools at effect strength lam.

    Direct keeps its measured pools. The VS arm's 'spread' pool rescales
    the pilot deviations; the 'slope', 'cos', 'ent_ab', and 'ent_ba' pools
    shift the Direct pool by lam times the seed-level offset.
    """
    def sl(arm):
        return slopes_fixed[arm][AXES].to_numpy().reshape(50, 9, 5)

    return {
        "direct": {
            "spread": scores_dir,
            "slope": sl("direct"),
            "cos": cos_pools["direct"][:, :, None],
            "ent_ab": ent_ab_pools["direct"][:, :, None],
            "ent_ba": ent_ba_pools["direct"][:, :, None],
        },
        "vs": {
            "spread": synth_var_pool(scores_dir, scores_vs, lam),
            "slope": synth_shift_pool(sl("direct"), sl("vs"), lam),
            "cos": synth_shift_pool(cos_pools["direct"][:, :, None],
                                    cos_pools["vs_weighted"][:, :, None], lam),
            "ent_ab": synth_shift_pool(ent_ab_pools["direct"][:, :, None],
                                       ent_ab_pools["vs_weighted"][:, :, None],
                                       lam),
            "ent_ba": synth_shift_pool(ent_ba_pools["direct"][:, :, None],
                                       ent_ba_pools["vs_weighted"][:, :, None],
                                       lam),
        },
    }


def synth_var_pool(pool_dir, pool_vs, lam):
    """Synthetic VS spread pool at effect strength lam."""
    dev = pool_vs - pool_vs.mean(axis=1, keepdims=True)
    v_s = pool_vs.var(axis=1, ddof=1)
    v_d = pool_dir.var(axis=1, ddof=1)
    target = lam * v_s + (1 - lam) * v_d
    scale = np.sqrt(target / np.maximum(v_s, 1e-12))
    return pool_vs.mean(axis=1, keepdims=True) + scale[:, None, :] * dev


def synth_shift_pool(pool_dir, pool_vs, lam):
    """Synthetic VS pool for centered values (slopes, cosine, entailment)."""
    off = pool_vs.mean(axis=1) - pool_dir.mean(axis=1)
    return pool_dir + lam * off[:, None, :]


def one_task_stats(rng, n, k, arm_d, arm_v):
    """Per-seed value arrays for one simulated task study.

    Seeds are resampled with repetition (shared across the arms, so the
    arms pair through the seed texts); chain draws are independent per arm.
    """
    idx = rng.integers(0, scores_dir.shape[0], n)
    st = {}
    for key in ("spread", "slope", "cos", "ent_ab", "ent_ba"):
        A, Bp = arm_d[key], arm_v[key]
        vd = A[idx[:, None], rng.integers(0, A.shape[1], (n, k))]
        vv = Bp[idx[:, None], rng.integers(0, Bp.shape[1], (n, k))]
        if key == "spread":
            st["spread_d"] = vd.var(axis=1, ddof=1).mean(axis=1)
            st["spread_v"] = vv.var(axis=1, ddof=1).mean(axis=1)
        elif key == "slope":
            m_d, m_v = vd.mean(axis=1), vv.mean(axis=1)
            st["slope_ax_d"], st["slope_ax_v"] = m_d, m_v
            st["md_d"] = np.abs(m_d).mean(axis=1)
            st["md_v"] = np.abs(m_v).mean(axis=1)
        else:
            st[f"{key}_d"] = vd.mean(axis=1).ravel()
            st[f"{key}_v"] = vv.mean(axis=1).ravel()
    return st


def uniform_spread(rng, n, k, pool):
    """Per-seed spread under the uniform arm's labeled scenario."""
    vals = pool[rng.integers(0, pool.shape[0], n)[:, None],
                rng.integers(0, pool.shape[1], (n, k))]
    return vals.var(axis=1, ddof=1).mean(axis=1)


TASK_SIGNS = {"H1_var": 1, "H2a": -1, "H2b_abs": -1, "H2b_fct": -1, "H3": 1}


def task_evidence(rng, n, k, arm_d, arm_v, uniform_pool):
    """One simulated task study's five tests and two interval criteria."""
    st = one_task_stats(rng, n, k, arm_d, arm_v)
    st["spread_u"] = uniform_spread(rng, n, k, uniform_pool)
    t = {}
    t["H1_var"] = paired_check(st["spread_v"] - st["spread_d"])
    t["H2a"] = paired_check(st["cos_v"] - st["cos_d"])
    t["H2b_abs"] = paired_check(st["slope_ax_v"][:, IA]
                                - st["slope_ax_d"][:, IA])
    t["H2b_fct"] = paired_check(st["slope_ax_v"][:, IF]
                                - st["slope_ax_d"][:, IF])
    t["asym"] = paired_check((st["ent_ab_v"] - st["ent_ab_d"])
                             - (st["ent_ba_v"] - st["ent_ba_d"]))
    t["equiv"] = paired_check(st["md_v"] - st["md_d"])
    t["H3"] = paired_check(st["spread_v"] - st["spread_u"])
    return t


def cell_verdicts(rng, design, lam, tasks_like_pilot, uniform_label, b=B):
    """Outcome shares over b simulated studies of one (design, scenario).

    tasks_like_pilot=True runs the other three tasks as independent copies
    ('all tasks like the pilot'); the alternative contributes 15 exact
    null p-draws ('only paraphrase'). uniform_label in {'direct', 'halfway',
    'vs'}: the control behaves like Direct (H3 power), halfway between the
    arms' spreads, or like the VS arm at the current effect strength (under
    which H3 is null).
    Returns per-outcome shares with Monte Carlo standard errors, plus the
    mean interval widths the equivalence and asymmetry rules live on.
    """
    n, k = design
    arms = build_arms(lam)
    uniform_pool = {
        "direct": arms["direct"]["spread"],
        "halfway": synth_var_pool(scores_dir, scores_vs, 0.5),
        "vs": arms["vs"]["spread"],
    }[uniform_label]
    cnt = defaultdict(int)
    equiv_hw, asym_hw, hw_abs, hw_fct, equiv_mean = [], [], [], [], []
    for _ in range(b):
        t = task_evidence(rng, n, k, arms["direct"], arms["vs"],
                          uniform_pool)
        p5 = [t[name][0] for name in TASK_SIGNS]
        if tasks_like_pilot:
            p20 = list(p5)
            for _ in range(3):            # the other three task copies
                t2 = task_evidence(rng, n, k, arms["direct"], arms["vs"],
                                   uniform_pool)
                p20 += [t2[name][0] for name in TASK_SIGNS]
        else:
            p20 = p5 + list(rng.random(15))
        adj = bh_adjust(np.array(p20))
        adj5 = dict(zip(TASK_SIGNS, adj[:len(TASK_SIGNS)]))
        ok = {}
        for name, sign in TASK_SIGNS.items():
            ok[name] = adj5[name] <= ALPHA and t[name][1] * sign > 0
            cnt[name] += ok[name]
            cnt[name + "_raw"] += t[name][1] * sign > 0
        asym_ok = t["asym"][1] - t["asym"][2] > 0
        eq_ok = (t["equiv"][1] + t["equiv"][2] < BOUND
                 and t["equiv"][1] - t["equiv"][2] > -BOUND)
        cnt["H2b_comp"] += ok["H2b_abs"] and ok["H2b_fct"] and asym_ok
        cnt["H1_comp"] += ok["H1_var"] and eq_ok
        cnt["equiv_pass"] += eq_ok
        cnt["asym_pass"] += asym_ok
        cnt["family_rej"] += bool((adj <= ALPHA).any())
        equiv_hw.append(t["equiv"][2])
        equiv_mean.append(t["equiv"][1])
        asym_hw.append(t["asym"][2])
        hw_abs.append(t["H2b_abs"][2])
        hw_fct.append(t["H2b_fct"][2])
    out = {}
    for name in ("H1_var", "H2a", "H2b_abs", "H2b_fct", "H3", "H2b_comp",
                 "H1_comp", "equiv_pass", "asym_pass", "family_rej"):
        sh = cnt[name] / b
        out[name] = sh
        out[name + "_se"] = math.sqrt(sh * (1 - sh) / b)
    out["equiv_hw_mean"] = float(np.mean(equiv_hw))
    out["equiv_mean"] = float(np.mean(equiv_mean))
    out["asym_hw_mean"] = float(np.mean(asym_hw))
    out["hw_abs_mean"] = float(np.mean(hw_abs))
    out["hw_fct_mean"] = float(np.mean(hw_fct))
    return out


# %% [markdown]
# ## Results at the planned size
#
# Six candidate designs under three effect strengths. All designs except
# 112x5 cost the same 450 transmitted chains per condition and task; 112x5
# costs 560 (+24 percent). Under the 'all tasks like the pilot' scenario
# the three other tasks run as copies of the paraphrase simulation with
# fresh draws; the 'only paraphrase' variant leaves those 15 tests null.
# Column meanings:
#
# - **H1 comp**: spread contrast found in the family AND the equivalence
#   interval inside the bound;
# - **H2b comp**: both per-axis tests found AND the asymmetry interval
#   pointing the same way;
# - **H2a**: the similarity contrast found;
# - **H3**: the control contrast found (control labeled as Direct);
# - **equiv**: the equivalence interval inside the bound;
# - **family**: share of simulated studies with at least one family
#   rejection; under no effect this row is the false alarm rate.

# %%
def pct(x):
    return f"{100 * x:.0f}%"


rows = []
for lam in LAMBDAS:
    label = "none" if lam == 0.0 else f"{lam:.0%} of pilot"
    for design in DESIGNS:
        r = cell_verdicts(
            np.random.default_rng(
                SEED + int(1000 * lam) + design[0] * 97 + design[1]),
            design, lam, tasks_like_pilot=True, uniform_label="direct")
        rows.append({
            "design": f"{design[0]}x{design[1]}",
            "chains/task": design[0] * design[1],
            "effect": label,
            "H1": pct(r["H1_comp"]),
            "H2b": pct(r["H2b_comp"]),
            "H2a": pct(r["H2a"]),
            "H3": pct(r["H3"]),
            "equiv": pct(r["equiv_pass"]),
            "family": pct(r["family_rej"]),
        })
results = pd.DataFrame(rows)
print(results.to_string(index=False))
results

# %% [markdown]
# Under no effect, the family false alarm sits below the intended 5
# percent at every design (2 to 3 percent) and no preregistered test
# fires. At the pilot's effect, the spread, similarity, and control
# contrasts are found in every simulated study at every design, and the
# equivalence rule passes everywhere: its interval is two orders of
# magnitude tighter than the bound. The per-axis drift component (H2b) is
# the lowest-power component, and its power grows with the number of seeds: from 55
# percent at 50x9 to 99 percent at 150x3, with 88 percent at the planned
# 90x5.
#
# One estimand note. The equivalence rule works on the per-seed drift
# differences (the paper's descriptive drift numbers use the pooled
# signed-per-axis aggregation, which is smaller: 0.00015 axis units per
# hop at the pilot). The rule's own value at the pilot effect is the
# printed mean below; both sit far inside the 0.005 bound.

# %%
pilot_design = [r for r in rows if r["design"] == "90x5"]
for row in pilot_design:
    print(row)

# detailed interval widths at 90 x 5, pilot effect
r90 = cell_verdicts(np.random.default_rng(SEED + 5), (90, 5), 1.0,
                    tasks_like_pilot=True, uniform_label="direct")
print("\n90x5, pilot effect: "
      f"H1 comp {pct(r90['H1_comp'])}, H2b comp {pct(r90['H2b_comp'])}, "
      f"H2a {pct(r90['H2a'])}, H3 {pct(r90['H3'])}, "
      f"equiv {pct(r90['equiv_pass'])}")
print(f"equivalence interval: mean {r90['equiv_mean']:.2e} units/hop, "
      f"half-width {r90['equiv_hw_mean']:.2e} "
      f"(bound {BOUND:.3f}, factor {BOUND / r90['equiv_hw_mean']:.0f} above)")
print(f"asymmetry interval: half-width {r90['asym_hw_mean']:.3f}")
print(f"per-axis drift intervals: abstract {r90['hw_abs_mean']:.2e}, "
      f"factual {r90['hw_fct_mean']:.2e} units/hop")

# 112 x 5 (+24% cost): what the extra chains buy at half effect
r112 = cell_verdicts(np.random.default_rng(SEED + 6), (112, 5), 0.5,
                     tasks_like_pilot=True, uniform_label="direct")
print(f"\n112x5, half effect:   "
      f"H1 comp {pct(r112['H1_comp'])}, H2b comp {pct(r112['H2b_comp'])}, "
      f"H2a {pct(r112['H2a'])}, H3 {pct(r112['H3'])}, "
      f"equiv {pct(r112['equiv_pass'])}")

# %% [markdown]
# ## The control contrast under its labeled scenarios
#
# H3's power rests entirely on where the Multi-k uniform control's spread
# sits, and the pilot measured no control arm. Three labeled scenarios
# bracket it: the control behaves like Direct, halfway between the arms'
# spread levels, or like the VS arm (under which H3 must read as null; its
# share below is the scenario's false alarm).

# %%
h3_rows = []
for lam in (0.0, 0.5, 1.0):
    for ul in ("direct", "halfway", "vs"):
        r = cell_verdicts(
            np.random.default_rng(SEED + 900 + int(1000 * lam)),
            (90, 5), lam, tasks_like_pilot=True, uniform_label=ul)
        h3_rows.append({
            "effect": ("none" if lam == 0.0 else f"{lam:.0%} of pilot"),
            "control spread at": ul,
            "H3 found": pct(r["H3"]),
        })
print(pd.DataFrame(h3_rows).to_string(index=False))

# %% [markdown]
# ## Sensitivity checks
#
# Three checks protect the conclusions: a subgroup stand-in for the genre
# cells, the grid-averaged pools in place of the fixed setting, and the
# chain-score discreteness the empirical pools carry.

# %% [markdown]
# **Genre-cell stand-in.** The pilot pools have no verified genre labels,
# so 30 resampled seeds (the size of one genre cell in the confirmatory
# corpus) stand in for a single cell's analysis. This is a hypothetical
# subgroup, not a genre-specific result.

# %%
r30 = cell_verdicts(np.random.default_rng(SEED + 777), (30, 5), 1.0,
                    tasks_like_pilot=True, uniform_label="direct")
print(f"30 seeds x 5 chains, pilot effect: H1 comp {pct(r30['H1_comp'])}, "
      f"H2b comp {pct(r30['H2b_comp'])}, H2a {pct(r30['H2a'])}, "
      f"equiv {pct(r30['equiv_pass'])}")

# %% [markdown]
# **Grid-averaged pools.** The confirmatory study fixes one decoding
# setting per model; the pilot's nine settings average to a smaller spread
# ratio (8.52 against the fixed setting's 10.73). Two questions follow, and
# each gets its own simulated study on the settings pools:
#
# - does the spread test stay well powered when the seed-level spread is
#   the mean over the nine settings rather than the fixed setting's?
# - does the drift equivalence rule still pass when the
#   per-seed drift is the mean over the nine settings?

# %%
def cfg_averaged(design, lam, b, measure="spread"):
    """Grid-averaged spread power or drift-equivalence pass rate at a design.

    measure 'spread': per-seed value = mean over settings of the
    mean-over-axes variance of k drawn chains; verdict = spread contrast
    found (unadjusted here; the family scenario adds nothing, the fixed-
    setting pass rates above already carry it). measure 'drift': per-seed
    value = mean over settings of the mean absolute drift; rule = the 95%
    interval of the paired differences inside the bound.
    """
    n, k = design
    pd_ = pools_per_config(load_axis("axis_cumulative_long.csv"))
    pv_ = pools_per_config(load_axis("vs_weighted/axis_cumulative_long.csv"))
    if measure == "drift":
        s_d = slopes_per_config(pd_, load_axis("axis_cumulative_long.csv"))
        s_v = slopes_per_config(pv_, load_axis(
            "vs_weighted/axis_cumulative_long.csv"))
        slope_pools_dir, slope_pools_vs = s_d, s_v
    rng = np.random.default_rng(SEED + 4242 + int(1000 * lam))
    found = eq_pass = 0
    hw_all = []
    for _ in range(b):
        idx = rng.integers(0, 50, n)
        v_vals, d_vals = [], []
        for cfg in pd_:
            if measure == "spread":
                wd = pd_[cfg][idx[:, None], rng.integers(0, 9, (n, k))]
                wv = synth_var_pool(pd_[cfg], pv_[cfg], lam)
                wv = wv[idx[:, None], rng.integers(0, 9, (n, k))]
                v_vals.append(wv.var(axis=1, ddof=1).mean(axis=1))
                d_vals.append(wd.var(axis=1, ddof=1).mean(axis=1))
            else:
                sdv = synth_shift_pool(slope_pools_dir[cfg],
                                       slope_pools_vs[cfg], lam)
                sdv = sdv[idx[:, None], rng.integers(0, 9, (n, k))]
                sdd = slope_pools_dir[cfg][idx[:, None],
                                           rng.integers(0, 9, (n, k))]
                v_vals.append(np.abs(sdv.mean(axis=1)).mean(axis=1))
                d_vals.append(np.abs(sdd.mean(axis=1)).mean(axis=1))
        diff = np.mean(v_vals, axis=0) - np.mean(d_vals, axis=0)
        p, mean, hw = paired_check(diff)
        hw_all.append(hw)
        if measure == "spread":
            found += (p <= ALPHA) and mean > 0
        else:
            eq_pass += mean + hw < BOUND and mean - hw > -BOUND
    if measure == "spread":
        return {"found": found / b, "hw": float(np.mean(hw_all))}
    return {"equiv_pass": eq_pass / b, "hw": float(np.mean(hw_all))}


def slopes_per_config(score_pools, df):
    """Per-config chain slope pools for the settings grid."""
    sl = pool_slopes_all(df)
    return {cfg: sl[cfg] for cfg in score_pools}


def pool_slopes_all(df):
    """Chain slope pools per config: {config: (50, 9, 5)}."""
    out = {}
    for cfg in sorted(df["config"].unique()):
        s = chain_slopes(df.loc[df["config"] == cfg])
        wide = s.pivot_table(index=["msg_id", "chain"], values=AXES)
        wide = wide.reorder_levels(["msg_id", "chain"]).sort_index()
        sh = wide.index.get_level_values("msg_id").nunique()
        assert wide.shape[0] == sh * 9, f"incomplete slope grid in {cfg}"
        out[cfg] = (wide.to_numpy().astype(float)
                    .reshape(sh, 9, len(AXES)))
    return out


for lam in (0.0, 1.0):
    tag = "none" if lam == 0.0 else "pilot"
    r = cfg_averaged((90, 5), lam, 1000, measure="spread")
    print(f"grid-averaged spread, effect {tag}: found {pct(r['found'])}; "
          f"contrast half-width {r['hw']:.2e}")
    r = cfg_averaged((90, 5), lam, 1000, measure="drift")
    print(f"grid-averaged drift, effect {tag}: equivalence "
          f"{pct(r['equiv_pass'])}; interval half-width {r['hw']:.2e}")

# %% [markdown]
# **Endpoint discreteness.** Chain endings repeat: under Direct, the nine
# chains of a seed often converge to one paraphrase and repeat it, so that
# seed's measured spread can be zero. The empirical pools carry this
# structure into the simulation; the two counts below report how much of it
# exists.

# %%
deg_d = (scores_dir.var(axis=1, ddof=1) == 0).mean()
deg_v = (scores_vs.var(axis=1, ddof=1) == 0).mean()
print(f"Direct (seed, axis) cells with zero spread across the 9 chains: "
      f"{pct(deg_d)}")
print(f"VS (seed, axis) cells with zero spread: {pct(deg_v)}")

# %% [markdown]
# # What the preregistration takes from this notebook
#
# **The planned size carries the preregistered decision rules.** At 90
# seed texts and 5 chains, the pilot-sized spread, similarity, and control
# contrasts are found in every simulated study, the equivalence rule
# passes with its interval about a hundred times inside the bound, and the
# per-axis drift component passes with a comfortable margin (around 90
# percent). Calibration rests on the same machinery: under no effect, the
# family false alarm stays at 2 to 3 percent, below the intended 5.
#
# **The lowest-power component is the per-axis drift test.** Per-axis power grows with
# seeds, not chains: 150x3 reaches 99 percent on the per-axis component at
# the pilot effect, against 88 percent at 90x5. The locked design keeps
# the per-chain depth; the comparison table states what a budget-neutral
# or slightly costlier reorder would move.
#
# **The control contrast is a scenario claim.** H3's power holds near 100
# percent when the Multi-k uniform control's spread behaves like Direct,
# and the halfway and VS-like scenarios bracket what it would take to
# overturn a positive verdict; the prereg text quotes the control-scenario
# table, not a single number.
#
# **Honest constraints.** The pools come from one small model under the
# paraphrase task; the other models and tasks enter as labeled copies. The
# simulated chances assume seed texts resample like the pilot's; the genre
# cells are treated by the 30-seed stand-in, which the subgroup block
# reports (the per-axis component sits near 21 percent there, so no
# within-cell per-axis verdict claims). Simulation uncertainty is about
# one percentage point on each printed rate at 2,000 studies.

# %%
results.to_csv(DATA_DIR / "pub" / "power_sim_results.csv", index=False)
print("wrote", DATA_DIR / "pub" / "power_sim_results.csv")