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
# # How big does the study need to be?
#
# This notebook justifies the planned study size for the preregistration.
# The planned design is 90 seed texts, 5 chains per text, and 200 hops per
# chain, for each of four tasks and four conditions.
#
# **Is that enough for the two preregistered effects?**
#
# The first prediction: verbalized sampling (VS), the practice of asking a
# model for several candidate texts with stated probabilities, keeps chains
# diverse (measured as the variance of the last hops). In the pilot, the
# variance under VS was 8.5 times the variance under Direct. The second
# prediction: VS transmits less of what the seed said (measured as the
# similarity of the last hop to the seed text).
#
# We answer the question by simulation in three steps. 
# 1. We take the variance of the 50 pilot texts as the best available guess for how much texts really differ.
# 2. We note that measuring a spread with 5 chains gives an imperfect estimate, and that the size of this imperfection follows a known statistical rule, which we use directly.
# 3. We pretend to run the planned study many times, and we count how often it would find a difference that is really there.
#
# The short answer: the planned design finds the pilot-sized diversity
# effect in essentially every simulated run, even when we look at one kind
# of text alone. It would still find, with a nine in ten chance, a diversity
# effect more than five times weaker than the one the pilot measured. The
# similarity effect is found just as easily. We checked smaller designs at
# the same total workload, and they find weaker effects less often, not
# more.

# %%
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(
    __import__("os").environ.get(
        "DATA_DIR", str(Path.home() / "Dropbox/data_export/echo-chains")
    )
)
AXES = [
    "valence_neg_to_pos",
    "tone_neutral_to_intense",
    "moderate_to_extreme",
    "factual_to_narrative",
    "abstract_to_concrete",
]

# %% [markdown]
# ## The measure
#
# A transmission chain starts from one seed text and retells it 200 times.
# Each retelling is one hop. At the end of a chain we score the final text on
# five semantic scales, for example how factual or how narrative the language
# reads. For one seed text, the chains do not all end in the same place. The
# spread of their endings is the variance of these scores across the chains
# of that seed, averaged over the five scales. A larger spread means the
# chains stayed more diverse.
#
# The preregistered comparison is simple. Run several chains on the same
# seed under Direct prompting, and run several more under VS. Compare the
# two spreads on the same seed. A real effect means the VS spread is larger,
# on the same texts, again and again.

# %%
def load_seed_spreads(rel_path):
    """Per seed, one variance value per scale. Shape: (50 seed texts, 5 scales).

    The pilot recorded the spread of each seed in each of nine decoding
    settings. We average over the settings, as the confirmatory study fixes
    one setting. The values are variances.
    """
    fp = DATA_DIR / "data" / "echodrift_stats" / rel_path
    df = pd.read_csv(fp)
    per_seed = df.groupby("msg_id")[AXES].mean()
    return per_seed[AXES].to_numpy()


spreads_direct = load_seed_spreads("sigma_at_hop200.csv")
spreads_vs = load_seed_spreads("vs_weighted/sigma_at_hop200.csv")

mean_direct = spreads_direct.mean()
mean_vs = spreads_vs.mean()
ratio_pilot = mean_vs / mean_direct
print(f"pilot seeds measured: {spreads_direct.shape[0]}")
print(f"average spread, Direct:  {mean_direct:.5f}")
print(f"average spread, VS:      {mean_vs:.5f}")
print(f"ratio:                   {ratio_pilot:.1f}")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(np.log10(spreads_direct.mean(axis=1)), bins=20, alpha=0.6, label="Direct")
ax.hist(np.log10(spreads_vs.mean(axis=1)), bins=20, alpha=0.6, label="VS")
ax.set_xlabel("spread of chain endings (log scale)")
ax.set_ylabel("number of seed texts")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The spread differs a lot from text to text. Some texts lead all their
# chains to almost the same ending, others keep the chains apart. This is
# exactly why the study needs many seed texts: a single text tells us little,
# but the average over many texts is stable.

# %% [markdown]
# ## What five chains can measure
#
# The spread of one seed is measured from 5 chain endings. Such a measured
# variance is imperfect in a known way: if a seed's true variance is V, the
# measured variance behaves like V times a standard statistical rescaling
# (a chi-square draw with one degree of freedom lost per chain, divided by
# the number of chains). Sometimes the measured spread comes out too large,
# sometimes too small. We put that exact rule into the simulation, so no
# assumption about the error is smuggled in.
#
# Before simulating the whole study, one sanity check. If we rank the seed
# texts by their measured spread, how well does that ranking agree with the
# ranking by their true spread? We measure agreement with the rank
# correlation, a number that is 1 when two rankings agree exactly and 0 when
# they share nothing.

# %%
rng = np.random.default_rng(42)
rhos = []
for rep in range(1000):
    measured = (spreads_vs
                * rng.chisquare(4, size=spreads_vs.shape) / 4).mean(axis=1)
    true_rank = pd.Series(spreads_vs.mean(axis=1)).rank()
    rho = pd.Series(measured).rank().corr(true_rank)
    rhos.append(rho)
print(f"rank agreement of a 5-chain spread measure with the true ranking: "
      f"{np.mean(rhos):.2f} on average (1000 simulated rounds)")

# %% [markdown]
# A value of 0.86 means the 5-chain measure recovers which texts keep
# chains apart and which do not, almost perfectly. This matters for parts of
# the analysis that look at individual texts rather than the average.

# %% [markdown]
# ## Simulating the planned study
#
# One simulated study works like this.
#
# 1. Pick 90 seed texts. We measured 50, so the simulation picks the 90 with
#    repetition from the 50, which reuses measured spreads instead of
#    inventing them.
# 2. For each picked text, draw what 5 chains would measure, once under
#    Direct and once under VS. The two arms share the same text, so a text
#    that is naturally wide is wide in both arms.
# 3. Compute the average difference between the VS spread and the Direct
#    spread across the 90 texts. Then apply the simplest possible test: the
#    difference counts as found when it exceeds roughly twice its own
#    typical variation, the usual two-sided 5 percent rule.
#
# We repeat this 2000 times. The share of runs in which the study finds the
# difference is the chance of finding it. We also check the same procedure
# inside one kind of text alone, because the corpus has three kinds and each
# kind gets 30 texts.

# %%
def chance_of_finding(spreads_direct, spreads_vs, n_texts, chains,
                      reps, rng, ratio=1.0):
    """Share of simulated studies that find the VS-minus-Direct difference.

    Each scale's spread is measured with the given number of chains and the
    known measurement rule, then the five scales are averaged, which mirrors
    the preregistered measure. The optional ratio sets how large the average
    effect is, as the ratio of the VS spread to the Direct spread. The pilot
    measured 8.5. We study smaller values because the confirmatory study
    uses four models, four tasks, and human-written seed texts, where the
    effect may be weaker.
    """
    n_pilot = spreads_direct.shape[0]
    scale = ratio / ratio_pilot
    vs_scenario = spreads_vs * scale
    found = 0
    for _ in range(reps):
        idx = rng.integers(0, n_pilot, size=n_texts)
        vs_meas = (vs_scenario[idx]
                   * rng.chisquare(chains - 1, size=(n_texts, len(AXES)))
                   / (chains - 1)).mean(axis=1)
        dir_meas = (spreads_direct[idx]
                    * rng.chisquare(chains - 1, size=(n_texts, len(AXES)))
                    / (chains - 1)).mean(axis=1)
        d = vs_meas - dir_meas
        se = np.std(d, ddof=1) / math.sqrt(n_texts)
        if abs(d.mean()) > 1.96 * se:
            found += 1
    return found / reps


rng = np.random.default_rng(7)
chance_all = chance_of_finding(spreads_direct, spreads_vs, 90, 5, 2000, rng,
                               ratio=ratio_pilot)
chance_one_kind = chance_of_finding(spreads_direct, spreads_vs, 30, 5, 2000, rng,
                                    ratio=ratio_pilot)
print(f"chance of finding the pilot-sized effect, all 90 texts:      "
      f"{chance_all:.0%}")
print(f"chance of finding it inside one kind of text (30 texts):     "
      f"{chance_one_kind:.0%}")

# %% [markdown]
# ## What if the real effect is weaker than in the pilot?
#
# The pilot used one small model and one task. The effect may be weaker for
# the newer models in the panel, for the other tasks, and for the
# human-written seed texts. So we lower the assumed effect and repeat. A
# ratio of 1 means no effect at all, which shows the false alarm rate. A
# ratio of 8.5 is the pilot result.

# %%
scenarios = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, ratio_pilot]
rows = []
for ratio in scenarios:
    c_all = chance_of_finding(spreads_direct, spreads_vs, 90, 5, 2000, rng,
                              ratio=ratio)
    c_kind = chance_of_finding(spreads_direct, spreads_vs, 30, 5, 2000, rng,
                               ratio=ratio)
    rows.append((f"{ratio:.1f}", f"{c_all:.0%}", f"{c_kind:.0%}"))
table = pd.DataFrame(rows, columns=[
    "assumed effect (VS spread / Direct spread)",
    "chance, all 90 texts",
    "chance, one kind of text (30 texts)",
])
print(table.to_string(index=False))
table

# %% [markdown]
# The planned design finds a doubling of the spread with almost certainty,
# even when we look at one kind of text alone, and it finds a 50 percent
# larger spread with a nine in ten chance across all texts. When there is no
# effect, the false alarm rate sits near the intended 5 percent (the
# simulation gave 6 and 8 percent). The study is therefore not oversized for
# the preregistered question, and it has a clear safety margin against
# weaker effects.

# %% [markdown]
# ## Why 90 texts and 5 chains, and not another split
#
# The total workload is the number of texts times the number of chains. The
# three designs below all sit at the same total workload, roughly 450 chains
# per condition and task. They differ in how that workload is spent.

# %%
designs = [(50, 9), (75, 6), (90, 5)]
rows = []
for n_texts, chains in designs:
    c_all = chance_of_finding(spreads_direct, spreads_vs, n_texts, chains,
                              2000, rng, ratio=1.5)
    c_kind = chance_of_finding(spreads_direct, spreads_vs,
                               max(1, round(n_texts / 3)), chains, 2000, rng,
                               ratio=1.5)
    rows.append((f"{n_texts} texts x {chains} chains", f"{c_all:.0%}",
                 f"{c_kind:.0%}"))
table = pd.DataFrame(rows, columns=[
    "design at the same total workload",
    "chance of finding a 50 percent larger spread, all texts",
    "chance inside one kind of text",
])
print(table.to_string(index=False))
table

# %% [markdown]
# At a fixed total workload, spending it on more seed texts beats spending
# it on more chains per text, because the average over texts is the harder
# thing to estimate. The planned 90 by 5 design finds weaker effects best
# while still measuring each text well, and the ranking check above showed
# the 5-chain spread measure still recovers which texts are wide and which
# are narrow. Deeper designs win a little on that ranking, shallower
# designs lose the per-text information almost entirely.

# %% [markdown]
# ## The second preregistered effect: does VS cost information?
#
# The spread result says little without its price tag. The paper's second
# preregistered prediction is that VS keeps chains diverse but transmits
# less of what the seed said. The measure is the last hop's similarity to
# the seed text, a score that is 1 when two texts say the same thing and
# falls as they drift apart. Direct prompting should keep this score high,
# VS should lower it.
#
# This part is a mean, not a variance, so the measurement logic from above
# simplifies: for one seed, the per-chain similarity scores are averaged,
# and the average of 5 chain values jitters around the seed's true value by
# an amount that follows directly from how much the chains disagree with
# each other in the pilot.

# %%
def load_seed_fidelity(rel_path, metric="cosine"):
    """Per seed, two numbers: the mean and the disagreement (variance across
    chains) of the last hop's similarity. Primary decoding setting of the
    pilot (temperature 0.8, top-p 0.5), as in the confirmatory study."""
    fp = DATA_DIR / "data" / "echodrift_stats" / rel_path
    df = pd.read_csv(fp, usecols=["config", "msg_id", "metric", "value"])
    df = df[(df["metric"] == metric) & (df["config"] == "temp0.8_topp0.5")]
    per_seed = df.groupby("msg_id")["value"].agg(["mean", "var"])
    return per_seed["mean"].to_numpy(), per_seed["var"].fillna(0.0).to_numpy()


fid_direct, wid_direct = load_seed_fidelity("metrics_last_hop_cumulative.csv")
fid_vs, wid_vs = load_seed_fidelity("vs_weighted/metrics_last_hop_cumulative.csv")
drop_pilot = fid_direct.mean() - fid_vs.mean()
print(f"last-hop similarity: Direct {fid_direct.mean():.3f}, "
      f"VS {fid_vs.mean():.3f}, drop {drop_pilot:.3f} (50 texts)")

# %%
def chance_of_finding_drop(fid_direct, wid_direct, fid_vs, wid_vs, n_texts,
                           chains, reps, rng, drop):
    """Chance of finding a similarity drop of the given size.

    Same procedure as the spread part, with one difference: the per-chain
    similarity is a simple measurement, so the seed's measured mean carries
    the chains' disagreement divided by the chain count, drawn as a normal
    error.
    """
    n_pilot = len(fid_direct)
    pilot_drop = fid_direct - fid_vs
    found = 0
    for _ in range(reps):
        idx = rng.integers(0, n_pilot, size=n_texts)
        vs_true = fid_direct[idx] - pilot_drop[idx] * (drop / drop_pilot)
        direct_true = fid_direct[idx]
        vs_meas = vs_true + rng.normal(0.0, np.sqrt(wid_vs[idx] / chains))
        direct_meas = direct_true + rng.normal(0.0,
                                               np.sqrt(wid_direct[idx] / chains))
        d = direct_meas - vs_meas
        se = np.std(d, ddof=1) / math.sqrt(n_texts)
        if abs(d.mean()) > 1.96 * se:
            found += 1
    return found / reps


rng = np.random.default_rng(11)
rows = []
for drop in [0.0, 0.05, 0.10, 0.15, drop_pilot]:
    c_all = chance_of_finding_drop(fid_direct, wid_direct, fid_vs, wid_vs,
                                   90, 5, 2000, rng, drop)
    c_kind = chance_of_finding_drop(fid_direct, wid_direct, fid_vs, wid_vs,
                                    30, 5, 2000, rng, drop)
    rows.append((f"{drop:.2f}", f"{c_all:.0%}", f"{c_kind:.0%}"))
table = pd.DataFrame(rows, columns=[
    "assumed similarity drop (VS below Direct)",
    "chance, all 90 texts",
    "chance, one kind of text (30 texts)",
])
print(table.to_string(index=False))
table

# %% [markdown]
# ## How well can the study certify that VS leaves the drift unchanged?
#
# The first hypothesis has a second part. Beyond raising the spread, VS is
# predicted to leave the drift unchanged: the texts should not slide along
# the scales, in the average across the five scales, by more than a
# preregistered bound of 0.005 scale units per hop. The preregistered rule
# requires the whole uncertainty interval of the measured difference
# between VS and Direct to fit inside that bound.
#
# Certification therefore rests on how wide that interval is. Its width
# comes from two sources: the texts differ in their drift, and the 5 chains
# of a seed measure the seed's drift with an error. Both can be read off
# the pilot. The pilot's own drift numbers are tiny. At the primary
# decoding setting, the drift difference measures about 0.0006 scale units
# per hop, roughly eight times smaller than the bound, so the interval fits
# inside the bound with room to spare.
#
# The directional part of the fidelity hypothesis (the per-scale drift
# toward more abstract and more general language) uses this same drift
# machinery broken out per scale, so the width computed here applies to
# those contrasts too.

# %%
def per_seed_drift(rel_path):
    """Per chain and per scale, the slope of the scale score on the hop
    number, then two numbers per seed: the mean absolute drift and its
    disagreement across chains (variance). Primary decoding setting of the
    pilot (temperature 0.8, top-p 0.5), as in the confirmatory study."""
    fp = DATA_DIR / "data" / "echodrift_stats" / rel_path
    df = pd.read_csv(fp, usecols=["config", "msg_id", "chain", "hop"] + AXES)
    df = df[df["config"] == "temp0.8_topp0.5"]
    df["hc"] = df["hop"] - df["hop"].mean()
    g = df.groupby(["msg_id", "chain"], sort=True)
    hc2 = float((df["hc"] ** 2).sum()) / g.ngroups   # same hop grid per chain
    for axis in AXES:
        df[axis] = (df[axis] - g[axis].transform("mean")) * df["hc"]
    slopes = g[AXES].sum() / hc2                       # (450 chains, 5 scales)
    chain_drift = slopes.abs().mean(axis=1)
    per_seed = chain_drift.groupby(level="msg_id").agg(["mean", "var"])
    return per_seed["mean"].to_numpy(), per_seed["var"].fillna(0.0).to_numpy()


drift_direct, w_drift_direct = per_seed_drift("axis_cumulative_long.csv")
drift_vs, w_drift_vs = per_seed_drift("vs_weighted/axis_cumulative_long.csv")
bound = 0.005
print(f"mean absolute drift per hop: Direct {drift_direct.mean():.6f}, "
      f"VS {drift_vs.mean():.6f} (primary setting, 50 texts)")
print(f"bound: {bound}")

# Interval half-width at the planned design (90 texts, 5 chains):
between = (drift_vs - drift_direct).var(ddof=1)
within = float(np.mean(w_drift_vs + w_drift_direct)) / 5  # chains per seed
half_width = 1.96 * np.sqrt(between / 90 + within / 90)
print(f"uncertainty interval half-width at 90 texts x 5 chains: "
      f"{half_width:.6f} scale units")
print(f"the bound allows {bound / half_width:.1f} times this width")

# %% [markdown]
# ## What this means for the preregistration
#
# The preregistered size of 90 seed texts and 5 chains is justified on three
# grounds.
#
# 1. It finds the pilot-sized effect with certainty in every simulated run,
#    overall and inside each kind of text.
# 2. It keeps a good chance of finding much weaker effects. A 50 percent
#    larger spread is found with a nine in ten chance across all texts and a
#    six in ten chance inside one kind of text, and a doubling is found
#    almost always in both. The false alarm rate stays near the intended 5
#    percent when there is no effect.
# 3. At the same total workload, it is the best available split between
#    breadth across texts and depth per text, and the 5-chain spread measure
#    still ranks the texts usefully (rank agreement 0.86), which the
#    analyses of individual texts rely on.
#
# The second preregistered effect gets the same verdict. The similarity
# drop is found in every simulated run, even at one fifth of the pilot's
# size and even inside one kind of text alone (99 percent there). And the
# drift equivalence claim is comfortably within reach: at the planned
# design, the uncertainty interval around the measured drift difference is
# over 70 times narrower than the preregistered bound allows, and the
# pilot's own drift difference sits about eight times inside the bound.
#
# Two caveats belong in the preregistration.
#
# The check reuses measured spreads and fidelities from 50 pilot texts of
# one small model under the paraphrase task. It assumes the other texts and
# tasks behave similarly, which is the standard assumption for such a check
# and not a guarantee.
#
# The fidelity measure brings an extra source of variation: the chains of
# one seed disagree in how much of the seed survives, so the per-text
# similarity scores have a real spread of their own. The simulation carries
# that spread, read off the pilot, and with it the design still finds even
# small similarity drops.
#
# The simulated test is the simplest possible analysis of each average. The
# preregistered analysis pools information across texts and scales in a
# stricter way, which can only improve on this check. So the chances
# reported here are, if anything, a floor.
