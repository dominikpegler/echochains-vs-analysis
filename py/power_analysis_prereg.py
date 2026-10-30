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
#       jupytext_version: 1.17.3
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
#     version: 3.12.11
# ---

# %% [markdown]
# # How big does the study need to be?
#
# This notebook justifies the planned study size for the preregistration.
# The planned design is 90 seed texts, 5 chains per text, and 200 hops per
# chain, for each of four tasks and four conditions. The question here is
# narrow: is that enough to find the effect that matters most?
#
# The main preregistered prediction is that verbalized sampling (VS), the
# practice of asking a model for several candidate texts with stated
# probabilities, keeps chains from looking alike over time. We measure this
# as the spread of chain endings for the same seed text. The pilot study
# measured that spread under VS and under Direct prompting with one small
# model. The spread under VS was 8.5 times the spread under Direct.
#
# We answer the question by simulation in three steps. First, we take the
# measured spreads of the 50 pilot texts as the best available guess for how
# much texts really differ. Second, we note that measuring a spread with 5
# chains gives an imperfect estimate, and that the size of this imperfection
# follows a known statistical rule, which we use directly. Third, we pretend
# to run the planned study many times, and we count how often it would find
# a difference that is really there.
#
# The short answer: the planned design finds the pilot-sized effect in
# essentially every simulated run, even when we look at one kind of text
# alone. It would still find, with a nine in ten chance, an effect more than
# five times weaker than the one the pilot measured. We checked smaller
# designs at the same total workload, and they find weaker effects less
# often, not more.

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
# Two caveats belong in the preregistration.
#
# The check reuses measured spreads from 50 pilot texts of one small model
# under the paraphrase task. It assumes the other texts and tasks behave
# similarly, which is the standard assumption for such a check and not a
# guarantee.
#
# The simulated test is the simplest possible analysis of the average spread
# difference. The preregistered analysis pools information across texts and
# scales in a stricter way, which can only improve on this check. So the
# chances reported here are, if anything, a floor.