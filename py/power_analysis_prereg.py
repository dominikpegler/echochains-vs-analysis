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
AX_LABELS = ["valence", "tone", "extremity", "factual", "abstract"]
IA, IF = 4, 3      # H3 axes: IA = abstract-to-concrete, IF =
                   # factual-to-narrative (AXES positions)
CONFIG = "temp0.8_topp0.5"
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
slopes_dir_full = chain_slopes(axis_dir)
slopes_vs_full = chain_slopes(axis_vs)
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


fdir = config_drift(slopes_dir_full)
fvs = config_drift(slopes_vs_full)
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
# The eight preregistered tests of a task:
#
# - **Spread (H1).** Per seed and axis, the variance of the axis scores
#   across the seed's chains, averaged over the five axes. Direction: VS
#   above Direct.
# - **Similarity (H2 cosine).** Per seed, the mean last-hop cosine.
#   Direction: VS below Direct.
# - **Seed-to-hop entailment (H2).** Per seed, the mean last-hop
#   seed-to-hop entailment. Direction: VS below Direct.
# - **Hop-to-seed entailment (H2).** Per seed, the mean last-hop
#   hop-to-seed entailment. Direction: VS below Direct.
# - **Abstract-to-concrete drift (H3a).** Per seed, the signed
#   chain-average slope. Direction: below 0.
# - **Factual-to-narrative drift (H3a).** Per seed, the signed
#   chain-average slope. Direction: below 0.
# - **Entailment asymmetry (H3b).** Per seed, the seed-to-hop minus
#   hop-to-seed entailment difference. Direction: above 0.
# - **Control contrast (H4).** The spread estimator against the Multi-k
#   uniform control. Direction: VS above control.
#
# A task's eight tests enter the per-model Benjamini-Hochberg family over
# the four tasks (32 tests, false discovery rate 0.05). A test counts as
# found when the adjusted test rejects and the estimate points the
# preregistered way. H1 has one test. H2 carries three conjunctive criteria
# and is supported only when all of its criteria are found. H3 has two
# components, each with its own verdict and no verdict over both together:
# H3a carries two conjunctive criteria (the two drift-axis tests), and H3b
# carries one criterion (the entailment asymmetry). The three axes without
# a directional prediction
# (valence, tone intensity, extremity) carry no confirmatory test; the
# simulation reports their drift estimates and paired intervals as
# descriptive outputs.

# %% [markdown]
# ## Scenario machinery
#
# The effect strength runs on one dial, $\lambda \in \{0, 0.5, 1\}$. At
# $\lambda = 0$ the VS arm reproduces the Direct pool seed by seed, so the
# whole 32-test family is null and the run measures false alarms. At
# $\lambda = 1$ the VS arm is the pilot's own measured values. At 0.5 the
# seed-level effect is halfway: synthetic VS chain values are the Direct
# pool plus half the seed-level VS-minus-Direct offset, and the spread
# scenario rescales the VS chains' within-seed deviations so each seed's
# variance interpolates between its Direct and pilot-VS values. Each seed
# keeps its own level throughout; nothing in the scenarios invents a
# distribution.
#
# The Multi-k uniform control has no pilot data. Three labeled scenarios
# stand in for it: the control behaves like Direct (power scenario for
# H4), halfway between the arms' spread levels, or like the VS arm at the
# current effect strength (under which H4 is null).

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
            st["slope_ax_d"], st["slope_ax_v"] = vd.mean(axis=1), vv.mean(axis=1)
        else:
            st[f"{key}_d"] = vd.mean(axis=1).ravel()
            st[f"{key}_v"] = vv.mean(axis=1).ravel()
    return st


def uniform_spread(rng, n, k, pool):
    """Per-seed spread under the uniform arm's labeled scenario."""
    vals = pool[rng.integers(0, pool.shape[0], n)[:, None],
                rng.integers(0, pool.shape[1], (n, k))]
    return vals.var(axis=1, ddof=1).mean(axis=1)


TASK_SIGNS = {"H1_var": 1, "H2_cos": -1, "H2_ent_ab": -1,
              "H2_ent_ba": -1, "H3_abs": -1, "H3_fct": -1,
              "H3_asym": 1, "H4": 1}
TESTS = list(TASK_SIGNS)


def task_evidence(rng, n, k, arm_d, arm_v, uniform_pool):
    """One simulated task study's eight preregistered tests."""
    st = one_task_stats(rng, n, k, arm_d, arm_v)
    st["spread_u"] = uniform_spread(rng, n, k, uniform_pool)
    t = {}
    t["H1_var"] = paired_check(st["spread_v"] - st["spread_d"])
    t["H2_cos"] = paired_check(st["cos_v"] - st["cos_d"])
    t["H2_ent_ab"] = paired_check(st["ent_ab_v"] - st["ent_ab_d"])
    t["H2_ent_ba"] = paired_check(st["ent_ba_v"] - st["ent_ba_d"])
    t["H3_abs"] = paired_check(st["slope_ax_v"][:, IA]
                                - st["slope_ax_d"][:, IA])
    t["H3_fct"] = paired_check(st["slope_ax_v"][:, IF]
                                - st["slope_ax_d"][:, IF])
    t["H3_asym"] = paired_check((st["ent_ab_v"] - st["ent_ab_d"])
                                 - (st["ent_ba_v"] - st["ent_ba_d"]))
    t["H4"] = paired_check(st["spread_v"] - st["spread_u"])
    return t, st


def cell_verdicts(rng, design, lam, tasks_like_pilot, uniform_label, b=B):
    """Outcome shares over b simulated studies of one (design, scenario).

    tasks_like_pilot=True runs the other three tasks as independent copies
    ('all tasks like the pilot'); the alternative contributes 24 exact
    null p-draws ('only paraphrase'). uniform_label in {'direct', 'halfway',
    'vs'}: the control behaves like Direct (H3 power), halfway between the
    arms' spreads, or like the VS arm at the current effect strength (under
    which H3 is null).
    Returns per-outcome shares with Monte Carlo standard errors, plus the
    mean interval half-widths of every component estimate. Pass rates count
    the registered per-task verdict of the paraphrase task copy (the other
    task copies contribute their tests to the same family).
    """
    n, k = design
    arms = build_arms(lam)
    uniform_pool = {
        "direct": arms["direct"]["spread"],
        "halfway": synth_var_pool(scores_dir, scores_vs, 0.5),
        "vs": arms["vs"]["spread"],
    }[uniform_label]
    cnt = defaultdict(int)
    hw = {name: [] for name in TESTS}
    hw_axis = {ax: [] for ax in AXES}
    for _ in range(b):
        t, st = task_evidence(rng, n, k, arms["direct"], arms["vs"],
                              uniform_pool)
        p8 = [t[name][0] for name in TESTS]
        if tasks_like_pilot:
            p32 = list(p8)
            for _ in range(3):            # the other three task copies
                t2, _ = task_evidence(rng, n, k, arms["direct"], arms["vs"],
                                      uniform_pool)
                p32 += [t2[name][0] for name in TESTS]
        else:
            p32 = p8 + list(rng.random(24))
        adj = bh_adjust(np.array(p32))
        adj8 = dict(zip(TESTS, adj[:len(TESTS)]))
        ok = {}
        for name, sign in TASK_SIGNS.items():
            ok[name] = adj8[name] <= ALPHA and t[name][1] * sign > 0
            cnt[name] += ok[name]
            cnt[name + "_raw"] += t[name][1] * sign > 0
            hw[name].append(t[name][2])
        h2ok = (ok["H2_cos"] and ok["H2_ent_ab"] and ok["H2_ent_ba"])
        h3aok = ok["H3_abs"] and ok["H3_fct"]
        h3bok = ok["H3_asym"]
        cnt["H2"] += h2ok
        cnt["H3a"] += h3aok
        cnt["H3b"] += h3bok
        cnt["H1"] += ok["H1_var"]
        cnt["family_rej"] += bool((adj <= ALPHA).any())
        for j, ax in enumerate(AXES):
            hw_axis[ax].append(paired_check(st["slope_ax_v"][:, j]
                                            - st["slope_ax_d"][:, j])[2])
    out = {}
    for name in TESTS + ["H2", "H3a", "H3b", "H1", "family_rej"]:
        sh = cnt[name] / b
        out[name] = sh
        out[name + "_se"] = math.sqrt(sh * (1 - sh) / b)
        if name in hw:
            out[name + "_hw_mean"] = float(np.mean(hw[name]))
    for ax in AXES:
        out[f"drift_hw_{ax}"] = float(np.mean(hw_axis[ax]))
    return out


# %% [markdown]
# ## Results at the planned size
#
# Six candidate designs under three effect strengths. All designs except
# 112x5 cost the same 450 transmitted chains per condition and task; 112x5
# costs 560 (+24 percent). Under the 'all tasks like the pilot' scenario
# the three other tasks run as copies of the paraphrase simulation with
# fresh draws; the 'only paraphrase' variant leaves those 24 tests null.
# Column meanings:
#
# - **H1**: the variance contrast found;
# - **H2**: all three fidelity criteria found (cosine, seed-to-hop
#   entailment, hop-to-seed entailment);
# - **H3a**: both per-axis drift tests found (drift direction);
# - **H3b**: the entailment-asymmetry test found (loss mode); no verdict is
#   formed over H3a and H3b together;
# - **H4**: the control contrast found (control labeled as Direct);
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
            "H1": pct(r["H1_var"]),
            "H2": pct(r["H2"]),
            "H3a": pct(r["H3a"]),
            "H3b": pct(r["H3b"]),
            "H4": pct(r["H4"]),
            "family": pct(r["family_rej"]),
        })
results = pd.DataFrame(rows)
print(results.to_string(index=False))
results

# %% [markdown]
# One reporting note. Pass rates count the registered per-task verdict of
# the paraphrase task copy; the other three task copies contribute their
# tests to the same Benjamini-Hochberg family.

# %%
pilot_design = [r for r in rows if r["design"] == "90x5"]
for row in pilot_design:
    print(row)

# detailed rates and interval widths at 90 x 5, pilot effect
r90 = cell_verdicts(np.random.default_rng(SEED + 5), (90, 5), 1.0,
                    tasks_like_pilot=True, uniform_label="direct")
print("\n90x5, pilot effect: "
      f"H1 {pct(r90['H1_var'])}, H2 {pct(r90['H2'])}, H3a {pct(r90['H3a'])}, "
      f"H3b {pct(r90['H3b'])}, H4 {pct(r90['H4'])}, "
      f"family rejection {pct(r90['family_rej'])}")
print("interval half-widths, mean over simulated studies:")
print(f"  variance contrast (H1):      {r90['H1_var_hw_mean']:.2e}")
print(f"  cosine (H2):                 {r90['H2_cos_hw_mean']:.3f}")
print(f"  seed-to-hop entailment (H2): {r90['H2_ent_ab_hw_mean']:.3f}")
print(f"  hop-to-seed entailment (H2): {r90['H2_ent_ba_hw_mean']:.3f}")
print(f"  entailment asymmetry (H3):   {r90['H3_asym_hw_mean']:.3f}")
for ax, lab in zip(AXES, AX_LABELS):
    print(f"  drift, {lab:9s}: {r90[f'drift_hw_{ax}']:.2e} axis units/hop "
          f"({'H3a axis' if ax in (AXES[IA], AXES[IF]) else 'descriptive'})")

# %% [markdown]
# ## The control contrast under its labeled scenarios
#
# H4's power rests entirely on where the Multi-k uniform control's spread
# sits, and the pilot measured no control arm. Three labeled scenarios
# bracket it: the control behaves like Direct, halfway between the arms'
# spread levels, or like the VS arm (under which H4 must read as null; its
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
            "H4 found": pct(r["H4"]),
        })
print(pd.DataFrame(h3_rows).to_string(index=False))

# %% [markdown]
# ## Sensitivity checks
#
# Two checks protect the conclusions: a subgroup stand-in for the genre
# cells, and the grid-averaged pools in place of the fixed setting.

# %% [markdown]
# **Genre-cell stand-in.** The pilot pools have no verified genre labels,
# so 30 resampled seeds (the size of one genre cell in the confirmatory
# corpus) stand in for a single cell's analysis. This is a hypothetical
# subgroup, not a genre-specific result.

# %%
r30 = cell_verdicts(np.random.default_rng(SEED + 777), (30, 5), 1.0,
                    tasks_like_pilot=True, uniform_label="direct")
print(f"30 seeds x 5 chains, pilot effect: H1 {pct(r30['H1_var'])}, "
      f"H2 {pct(r30['H2'])}, H3a {pct(r30['H3a'])}, H3b {pct(r30['H3b'])}, "
      f"H4 {pct(r30['H4'])}")

# %% [markdown]
# **Grid-averaged pools.** The confirmatory study fixes one decoding
# setting per model; the pilot's nine settings average to a smaller spread
# ratio (8.52 against the fixed setting's 10.73). The check below asks:
# does the spread test stay well powered when the seed-level spread is
# the mean over the nine settings rather than the fixed setting's?

# %%
def cfg_averaged(design, lam, b):
    """Grid-averaged spread power at a design.

    Per-seed value = mean over settings of the mean-over-axes variance of
    k drawn chains; verdict = spread contrast found (unadjusted here; the
    family scenario adds nothing, the fixed-setting pass rates above
    already carry it).
    """
    n, k = design
    pd_ = pools_per_config(load_axis("axis_cumulative_long.csv"))
    pv_ = pools_per_config(load_axis("vs_weighted/axis_cumulative_long.csv"))
    rng = np.random.default_rng(SEED + 4242 + int(1000 * lam))
    found = 0
    hw_all = []
    for _ in range(b):
        idx = rng.integers(0, 50, n)
        v_vals, d_vals = [], []
        for cfg in pd_:
            wd = pd_[cfg][idx[:, None], rng.integers(0, 9, (n, k))]
            wv = synth_var_pool(pd_[cfg], pv_[cfg], lam)
            wv = wv[idx[:, None], rng.integers(0, 9, (n, k))]
            v_vals.append(wv.var(axis=1, ddof=1).mean(axis=1))
            d_vals.append(wd.var(axis=1, ddof=1).mean(axis=1))
        diff = np.mean(v_vals, axis=0) - np.mean(d_vals, axis=0)
        p, mean, hw = paired_check(diff)
        hw_all.append(hw)
        found += (p <= ALPHA) and mean > 0
    return {"found": found / b, "hw": float(np.mean(hw_all))}


for lam in (0.0, 1.0):
    tag = "none" if lam == 0.0 else "pilot"
    r = cfg_averaged((90, 5), lam, 1000)
    print(f"grid-averaged spread, effect {tag}: found {pct(r['found'])}; "
          f"contrast half-width {r['hw']:.2e}")

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
# seed texts and 5 chains, the pilot-sized H1 variance contrast, all three
# H2 fidelity criteria, and the H4 control contrast are found in every
# simulated study; H3a (drift direction, both axes) passes in 88 percent
# of simulated studies (90 percent in the detailed rerun) and H3b (loss
# mode, the entailment asymmetry) in 100 percent. Calibration: under no
# effect, the family false alarm is 2 percent
# at the planned size (1 to 4 percent across the candidate designs), below
# the intended 5.
#
# **The lowest-powered hypothesis is H3a (drift direction).** Its power
# grows with the number of seeds, not chains: 57 percent at 50x9, 79
# percent at 75x6, 88 percent at 90x5, 95 percent at 112x4, and 99
# percent at 150x3 at the pilot effect. The locked design keeps the per-chain
# depth; the design table states what a budget-neutral or slightly
# costlier reorder would move.
#
# **The control contrast is a scenario claim.** H4 holds near 100 percent
# when the Multi-k uniform control's spread behaves like Direct. With the
# control halfway between the arms, H3 is near 2 percent at half the
# pilot effect and 100 percent at the pilot effect; with a VS-like
# control it reads as the 2 percent null scenario. The prereg text quotes
# the control-scenario table, not a single number.
#
# **Descriptive intervals.** At the planned size and the pilot effect, the
# paired interval half-widths average 3.5e-3 for the variance contrast,
# 0.028 for cosine, 0.079 for seed-to-hop and 0.058 for hop-to-seed
# entailment, and 0.087 for the entailment asymmetry; the per-axis drift
# intervals average 7.97e-5 (tone intensity) to 1.56e-4 (valence) axis
# units per hop.
#
# **Honest constraints.** The pools come from one small model under the
# paraphrase task; the other models and tasks enter as labeled copies. The
# simulated chances assume seed texts resample like the pilot's; the genre
# cells are treated by the 30-seed stand-in, which the subgroup block
# reports (H3a sits near 25 percent and H3b near 81 percent there, so no
# within-cell verdict claims for either drift-direction component).
# Simulation uncertainty is about
# one percentage point on each printed rate at 2,000 studies.

# %%
results.to_csv(DATA_DIR / "pub" / "power_sim_results.csv", index=False)
print("wrote", DATA_DIR / "pub" / "power_sim_results.csv")
