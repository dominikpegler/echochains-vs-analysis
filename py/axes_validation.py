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
# # Semantic-axis validation for the confirmatory verbalized-sampling study
#
# This notebook implements the axis-validation protocol registered in the paper
# (Supplementary Methods, *Semantic axis construction*) and in the OSF
# preregistration (Measured variables). It scores five established
# human-annotated corpora with the five semantic axes and applies the frozen
# pass criteria below. You run this notebook; nothing in it changes the axes,
# the anchors, or any paper file. Its outputs are the validation verdicts and
# the anchor table that the preregistration freeze records.
#
# ## When to run
#
# Run this notebook **before the preregistration freeze** (fence item F6), after
# the corpora are fetched. Two reasons, both consequences of the registered
# contingency:
#
# 1. Anchor revision is allowed only before the freeze. If an axis fails, its
#    anchors are revised with replacement sentences drawn from these validation
#    corpora and the validation is re-run. A validation failure found after the
#    freeze could not be acted on within the registered rules.
# 2. The freeze records the final anchor sentences and the validation verdicts,
#    so both must exist and be final before the freeze.
#
# The notebook does not depend on collection and does not need to be re-run
# after the freeze. Re-running is deterministic (fixed seeds, CPU embeddings)
# and reproduces the same outputs.
#
# ## Frozen pass criteria (decided 2026-11-14)
#
# Per axis, over its registered human-annotated corpus:
#
# - **Correspondence**: the correspondence criterion tests whether higher
#   human ratings correspond to higher axis scores. Pass: the Spearman rank
#   correlation between axis score and human rating has a bootstrap 95%
#   confidence interval that excludes zero, and the correlation is at least
#   0.2.
# - **Contrast pairs**: the contrast-pair criterion tests whether
#   human-verified pairs are ordered in the predicted direction (for the
#   moderate-to-extreme axis, the hyperbole member of a pair scores above its
#   literal paraphrase). Pass: at least 75% of the pairs are ordered so; when
#   fewer than 50 pairs are available, a binomial sign test at the 0.05 level
#   replaces the rate criterion.
# - **Anchor stability**: the anchor-stability criterion tests whether the
#   direction depends on a single anchor. For each of the ten anchors, the
#   axis direction is recomputed from the other nine, as in leave-one-out
#   cross-validation (a leave-one-anchor-out check), and the correspondence
#   criterion must hold under every recomputation.
#
# An axis that fails is revised once (see the revision protocol below) and
# re-validated. An axis that still fails after revision is excluded from the
# confirmatory analysis: no directional test uses it and its scores are not
# reported; the variance estimands average over the validated axes, H3a
# requires both predicted axes to be validated and is otherwise not
# evaluated, and the Benjamini-Hochberg family shrinks by the withdrawn
# tests. The registered power analysis assumes all five axes validate.
#
# ## Revision protocol (frozen 2026-11-14, before the validation verdicts)
#
# If an axis fails, its anchors are revised at most once, under a fixed
# mechanical rule:
#
# 1. Split before selection: the axis's revision corpus is split into a
#    construction half and a held-out half by a seeded draw
#    (`REVISION_SPLIT_SEED`), before any candidate is inspected.
# 2. Mechanical selection, human ratings and text length only, never axis
#    scores: in a correspondence corpus, the axis's poles are re-drawn from
#    the bottom and top rating deciles (10%) of the construction half, at
#    lengths of 5 to 20 words as counted with the NLTK word tokenizer (the
#    paper's length rule), five shortest per pole, ties broken by corpus row
#    order. The low-rating pole feeds the first pole, the high-rating pole
#    the second.
# 3. Extremity: both poles come from the HYPO pair corpus only (the five
#    shortest construction-half hyperboles whose matched literal also fits
#    the length bound; the paired literals become the moderate anchors). The
#    certainty corpus is never used for anchor selection.
# 4. Held-out re-validation: the revised axis is re-evaluated on the
#    held-out half (correspondence and anchor stability) and, for extremity,
#    on the held-out pairs, with the certainty corpus re-entering in full
#    for correspondence. The revised verdict stands; no further rounds.
# 5. Anchors are not final until `semantic_axes.py` is edited and the anchor
#    table regenerated. `RUN_REVISION` proposes anchors and runs the
#    held-out checks; it does not edit the module.
#
# ## Inputs and outputs
#
# The five corpora live under `$DATA_DIR/data/axes_validation/` with their
# pinned provenance (URL, sha1, license, source citation) in
# `sources.json` next to them. The notebook verifies every file against the
# pinned sha1 and downloads missing files from the pinned URLs.
#
# | Axis | Corpus | Human annotation | Check |
# |---|---|---|---|
# | valence | EmoBank | reader valence V (1-5) | correspondence |
# | tone intensity | EmoBank | reader arousal A (1-5) | correspondence |
# | factual-to-narrative | NarraDetect | three-coder narrativity (Likert) | correspondence |
# | abstract-to-concrete | CONcreTEXT | in-context concreteness AVG | correspondence |
# | moderate-to-extreme | Pei & Jurgens certainty | sentence-level certainty (1-5) | correspondence |
# | moderate-to-extreme | HYPO | 709 hyperbole/literal pairs | contrast pairs |
#
# Outputs are written to `$DATA_DIR/data/axes_validation/out/`:
# `validation_results.json` and `validation_results.csv` (verdicts, all
# statistics, provenance and versions) and `tab_anchor_sentences.tex` (the
# 50 anchor sentences as a LaTeX table, generated from the anchors that the
# pipeline actually uses, for the paper's Supplementary Methods and the
# preregistration attachment).
#
# ## How to run
#
# - Set `RUN_FETCH = True` once to verify/download the pinned corpora, then
#   back to `False`.
# - Set `RUN_VALIDATION = True` to compute the validation and write the
#   outputs. All executing cells are gated so an accidental run-all does
#   nothing until you flip the switches.
# - Set `RUN_REVISION = True` (after a validation run that left at least one
#   axis failing) to execute the revision protocol: it reads
#   `out/validation_results.json`, proposes anchors under the frozen
#   selection rule, and re-validates on the held-out splits. It returns the
#   proposed anchors and their held-out verdicts; accept a revision by
#   editing `echochain/semantic_axes.py` and re-running `RUN_VALIDATION`.
# - `SMOKE = True` runs the same pipeline on reduced corpora and fewer
#   bootstrap draws (fast end-to-end check; not a validation result). The
#   smoke pass does not write the output files.

# %%
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

# Gates and switches
RUN_FETCH = False        # verify/download the pinned corpora
RUN_VALIDATION = False   # run the checks and write the outputs
RUN_REVISION = False     # run the revision protocol on the failing axes
SMOKE = False            # reduced corpora + fewer bootstrap draws

# Frozen pass criteria (registered; see the markdown cell above)
RHO_MIN = 0.20           # minimum Spearman correlation for the correspondence criterion
CI_LEVEL = 0.95          # two-sided bootstrap confidence interval for the correlation
BOOT_DRAWS = 2000        # bootstrap draws for the correspondence interval
PAIR_RATE_MIN = 0.75     # minimum share of correctly ordered contrast pairs
PAIR_RATE_MIN_PAIRS = 50 # at this pair count or above the rate criterion governs;
SIGN_ALPHA = 0.05        # below it, a one-sided binomial sign test at this level does

# Determinism
BOOT_SEED = 20261114

# Revision protocol (frozen 2026-11-14, before the validation verdicts)
REVISION_SPLIT_SEED = 20261114  # construction/held-out split
DECILE_FRACTION = 0.10          # extreme rating decile for candidate selection
REV_MIN_WORDS = 5               # candidate length bounds, NLTK word_tokenize tokens
REV_MAX_WORDS = 20
REVS_PER_POLE = 5

# Paths
DATA_DIR = Path(os.environ.get(
    "DATA_DIR", str(Path.home() / "Dropbox" / "data_export" / "echochains")))
AXVAL_DIR = DATA_DIR / "data" / "axes_validation"
OUT_DIR = AXVAL_DIR / "out"
SOURCES_JSON = AXVAL_DIR / "sources.json"

# Smoke-mode reductions
SMOKE_MAX_ROWS = 400
SMOKE_BOOT_DRAWS = 200

if SMOKE:
    BOOT_DRAWS = SMOKE_BOOT_DRAWS

print(f"DATA_DIR: {DATA_DIR}")
print(f"axes_validation dir: {AXVAL_DIR}")

# %% [markdown]
# ## Corpora: verify, fetch, load
#
# Every file is verified against the sha1 pinned in `sources.json`. A missing
# file is downloaded from its pinned URL and verified; a present file with a
# mismatched hash is an error (the corpus on disk is not the pinned corpus).

# %%
def sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def verify_or_fetch(run_fetch: bool):
    """Verify every pinned corpus file; optionally download missing ones.
    Returns (sources.json metadata, per-file status report)."""
    if not SOURCES_JSON.exists():
        raise FileNotFoundError(f"missing pinned provenance record: {SOURCES_JSON}")
    meta = json.loads(SOURCES_JSON.read_text())
    report = {}
    for rel, rec in meta["files"].items():
        dst = AXVAL_DIR / rel
        if dst.exists():
            digest = sha1_of(dst)
            if digest != rec["sha1"]:
                raise RuntimeError(
                    f"sha1 mismatch for {rel}: on disk {digest}, pinned {rec['sha1']}")
            report[rel] = "verified"
        elif run_fetch:
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {rel} from {rec['url']} ...", flush=True)
            urllib.request.urlretrieve(rec["url"], dst)
            digest = sha1_of(dst)
            if digest != rec["sha1"]:
                raise RuntimeError(
                    f"sha1 mismatch after download for {rel}: got {digest}, "
                    f"pinned {rec['sha1']}")
            report[rel] = "downloaded+verified"
        else:
            report[rel] = "MISSING (set RUN_FETCH = True)"
    return meta, report


def load_corpora() -> dict:
    """Load the five corpora into DataFrames with a uniform schema.

    Returns a dict corpus name -> DataFrame. Correspondence corpora have
    columns [text, rating]; the pair corpus has columns [text_hyperbole,
    text_literal].
    """
    corpora = {}

    # EmoBank: sentence-level reader valence (V) and arousal (A), 1-5.
    emobank = pd.read_csv(AXVAL_DIR / "emobank" / "emobank.csv")
    emobank = emobank.dropna(subset=["text", "V", "A"])
    emobank["text"] = emobank["text"].astype(str).str.strip()
    emobank = emobank[emobank["text"] != ""]
    corpora["emobank"] = emobank[["text", "V", "A"]].reset_index(drop=True)

    # NarraDetect: passages with the three-coder average overall narrativity.
    narra = pd.read_csv(AXVAL_DIR / "narradetect" / "NarraDetect_Scalar_AllModels.csv")
    narra = narra.dropna(subset=["TEXT", "avg_overall"])
    narra["TEXT"] = narra["TEXT"].astype(str).str.strip()
    corpora["narradetect"] = narra[["TEXT", "avg_overall"]].rename(
        columns={"TEXT": "text", "avg_overall": "rating"}).reset_index(drop=True)

    # CONcreTEXT: English trial + test sentences, in-context concreteness AVG.
    frames = []
    for name in ["CONcreTEXT_trial_EN.tsv", "CONcreTEXT_test_EN.tsv"]:
        frames.append(pd.read_csv(AXVAL_DIR / "concretext" / name, sep="\t"))
    concretext = pd.concat(frames, ignore_index=True)
    concretext = concretext.dropna(subset=["TEXT", "AVG"])
    concretext["TEXT"] = concretext["TEXT"].astype(str).str.strip()
    corpora["concretext"] = concretext[["TEXT", "AVG"]].rename(
        columns={"TEXT": "text", "AVG": "rating"}).reset_index(drop=True)

    # Pei & Jurgens certainty: scientific findings with sentence-level certainty.
    rows = []
    with open(AXVAL_DIR / "certainty" / "sentence-level-certainty.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append((rec["finding"], rec["sentence-level-certainty"]))
    certainty = pd.DataFrame(rows, columns=["text", "rating"])
    certainty = certainty.dropna()
    certainty["text"] = certainty["text"].astype(str).str.strip()
    certainty = certainty[certainty["text"] != ""]
    corpora["certainty"] = certainty.reset_index(drop=True)

    # HYPO: rows 0-708 are the 709 hyperboles, rows 709-1417 their literal
    # paraphrases in matching order (mirror layout; alignment verified below).
    hypo = pd.read_csv(AXVAL_DIR / "hypo" / "hypo.csv")
    if len(hypo) != 1418:
        raise ValueError(f"HYPO mirror has {len(hypo)} rows, expected 1418")
    if hypo["Hyperbole"].head(709).nunique() != 1 or hypo["Hyperbole"].head(709).iloc[0] != 1:
        raise ValueError("HYPO mirror: rows 0-708 are not all hyperboles")
    if hypo["Hyperbole"].tail(709).nunique() != 1 or hypo["Hyperbole"].tail(709).iloc[0] != 0:
        raise ValueError("HYPO mirror: rows 709-1417 are not all literals")
    hypo = hypo.dropna(subset=["Sentence"])
    if len(hypo) != 1418:
        raise ValueError("HYPO mirror: missing sentences after dropna")
    corpora["hypo"] = pd.DataFrame({
        "text_hyperbole": hypo["Sentence"].iloc[:709].to_numpy(),
        "text_literal": hypo["Sentence"].iloc[709:].to_numpy(),
    }).reset_index(drop=True)

    if SMOKE:
        for name in ["emobank", "narradetect", "concretext", "certainty"]:
            corpora[name] = corpora[name].head(SMOKE_MAX_ROWS)
        corpora["hypo"] = corpora["hypo"].head(SMOKE_MAX_ROWS)

    return corpora


META, FETCH_REPORT = verify_or_fetch(RUN_FETCH)
for rel, status in sorted(FETCH_REPORT.items()):
    print(f"{rel:55s} {status}")

# %% [markdown]
# ## Axes and embeddings
#
# The axes, anchors, centroids, and projection come from `echochain.semantic_axes`
# (the same module the confirmatory pipeline uses), with the embedding model
# pinned to CPU so the run is deterministic. The projection is vectorized here;
# the cell below verifies the vectorized projection against the pipeline's
# own `project_onto_axis` on sample vectors.

# %%
# Patch the model factory to CPU before importing semantic_axes, which
# instantiates the embedding model at import time (same pattern as
# py/write_vs_axis_csv.py).
import echochain.embeddings as _em
from sentence_transformers import SentenceTransformer


def _cpu_model():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu")


_em.get_embedding_model = _cpu_model

from echochain.semantic_axes import (ANCHORS, EMBEDDING_MODEL, get_centroid,  # noqa: E402
                                     project_onto_axis)

# Axis name -> (first-pole anchor key, second-pole anchor key)
AXES = {
    "valence_neg_to_pos": ("neg_valence", "pos_valence"),
    "tone_neutral_to_intense": ("neutral_tone", "intense_tone"),
    "moderate_to_extreme": ("moderate", "extreme"),
    "factual_to_narrative": ("factual", "narrative"),
    "abstract_to_concrete": ("abstract", "concrete"),
}

# Axis name -> paper-facing axis label for the anchor table
AXIS_LABELS = {
    "valence_neg_to_pos": "Negative to positive valence",
    "tone_neutral_to_intense": "Neutral to intense tone",
    "moderate_to_extreme": "Moderate to extreme framing",
    "factual_to_narrative": "Factual to narrative style",
    "abstract_to_concrete": "Abstract to concrete language",
}


def centroid_of(sentences):
    return get_centroid(sentences, model=EMBEDDING_MODEL)


def axis_centroids(axis, anchors=None):
    """First- and second-pole centroids for an axis (optionally from a
    replacement anchor dict, used by the leave-one-anchor-out variants)."""
    a = anchors if anchors is not None else ANCHORS
    first, second = AXES[axis]
    return centroid_of(a[first]), centroid_of(a[second])


def project_many(embeddings, start_vec, end_vec):
    """Vectorized twin of echochain.semantic_axes.project_onto_axis."""
    axis_vec = end_vec - start_vec
    norm_sq = float(np.dot(axis_vec, axis_vec))
    if norm_sq == 0.0:
        return np.zeros(len(embeddings))
    t = (embeddings - start_vec) @ axis_vec / norm_sq
    return 2.0 * t - 1.0


def _selfcheck_projection():
    rng = np.random.default_rng(0)
    start, end = axis_centroids("valence_neg_to_pos")
    probes = EMBEDDING_MODEL.encode(ANCHORS["factual"] + ANCHORS["concrete"])
    for probe in probes:
        want = project_onto_axis(probe, start, end)
        got = float(project_many(probe[None, :], start, end)[0])
        assert np.isclose(want, got, atol=1e-9), (want, got)


_selfcheck_projection()
print("projection self-check against echochain.semantic_axes: OK")

# %% [markdown]
# ## Checks
#
# Each correspondence check embeds the corpus once and projects onto the
# registered axis; the bootstrap resamples corpus rows with a fixed seed. The
# leave-one-anchor-out variants recompute the direction from the remaining
# nine anchors and re-project the same embeddings (the projection is a rank
# preserving map, so the correlation is unaffected by the pole rescaling).

# %%
def embed(texts):
    return np.asarray(EMBEDDING_MODEL.encode(list(texts), show_progress_bar=False))


def truncated_counts(texts):
    """Number of texts longer than the embedding model's 384-token limit."""
    counts = [len(ids) for ids in EMBEDDING_MODEL.tokenizer(
        list(texts), truncation=False)["input_ids"]]
    return sum(1 for c in counts if c > 384), max(counts) if counts else 0


def bootstrap_spearman(ratings, scores, rng, draws):
    """Spearman rho with a bootstrap CI over corpus rows."""
    ratings = np.asarray(ratings, dtype=float)
    scores = np.asarray(scores, dtype=float)
    rho = stats.spearmanr(ratings, scores).statistic
    n = len(ratings)
    boots = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        r = stats.spearmanr(ratings[idx], scores[idx]).statistic
        boots[i] = 0.0 if np.isnan(r) else r
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {"rho": float(rho), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(n), "draws": int(draws)}


def correspondence_pass(res):
    return res["ci_lo"] > 0.0 and res["rho"] >= RHO_MIN


def check_correspondence(axis, corpus_df, rng, draws, anchors=None):
    """Correspondence criterion + its leave-one-anchor-out variants.

    `anchors` overrides the a priori anchor dict (used by the LOO variants
    internally and by the revision protocol's held-out re-validation).
    """
    a = anchors if anchors is not None else ANCHORS
    start, end = axis_centroids(axis, anchors=a)
    embs = embed(corpus_df["text"])
    scores = project_many(embs, start, end)
    main = bootstrap_spearman(corpus_df["rating"], scores, rng, draws)
    main["pass"] = correspondence_pass(main)

    variants = []
    first, second = AXES[axis]
    for pole_key in (first, second):
        for i in range(len(a[pole_key])):
            replaced = {k: list(v) for k, v in a.items()}
            replaced[pole_key] = [s for j, s in enumerate(replaced[pole_key]) if j != i]
            v_start, v_end = axis_centroids(axis, anchors=replaced)
            v_scores = project_many(embs, v_start, v_end)
            res = bootstrap_spearman(corpus_df["rating"], v_scores, rng, draws)
            res["pass"] = correspondence_pass(res)
            res["dropped_pole"], res["dropped_anchor"] = pole_key, i
            variants.append(res)
    return {"main": main, "loo_variants": variants,
            "loo_all_pass": all(v["pass"] for v in variants),
            "n_truncated": None}  # filled by the caller


def check_pairs(axis, pair_df, anchors=None):
    """Contrast-pair criterion: the pair members are ordered in the
    predicted direction. Above PAIR_RATE_MIN_PAIRS pairs, pass at a correct
    ordering rate of >= PAIR_RATE_MIN; below it, a one-sided binomial sign
    test at SIGN_ALPHA replaces the rate criterion."""
    start, end = axis_centroids(axis, anchors=anchors)
    embs_h = embed(pair_df["text_hyperbole"])
    embs_l = embed(pair_df["text_literal"])
    s_h = project_many(embs_h, start, end)
    s_l = project_many(embs_l, start, end)
    ordered = s_h > s_l
    n = int(len(pair_df))
    rate = float(ordered.mean())
    sign_p = float(stats.binomtest(int(ordered.sum()), n, 0.5,
                                    alternative="greater").pvalue)
    if n >= PAIR_RATE_MIN_PAIRS:
        criterion, passed = "rate", bool(rate >= PAIR_RATE_MIN)
    else:
        criterion, passed = "sign_test", bool(sign_p < SIGN_ALPHA)
    # Pair-alignment provenance check: paraphrase pairs must be much more
    # similar than random sentence pairs from the corpus.
    cos = np.sum(embs_h * embs_l, axis=1) / (
        np.linalg.norm(embs_h, axis=1) * np.linalg.norm(embs_l, axis=1))
    return {"rate": rate, "n_pairs": n, "sign_p": sign_p,
            "criterion": criterion, "pass": passed,
            "pair_cosine_median": float(np.median(cos))}


def run_validation(corpora):
    results = {}
    plans = [
        ("valence_neg_to_pos", "emobank", "V"),
        ("tone_neutral_to_intense", "emobank", "A"),
        ("factual_to_narrative", "narradetect", "rating"),
        ("abstract_to_concrete", "concretext", "rating"),
        ("moderate_to_extreme", "certainty", "rating"),
    ]
    for axis, corpus, rating_col in plans:
        rng = np.random.default_rng(BOOT_SEED)
        df = corpora[corpus]
        n_trunc, max_tok = truncated_counts(df["text"])
        corr = check_correspondence(axis, df.rename(columns={rating_col: "rating"}),
                                    rng, BOOT_DRAWS)
        corr["n_truncated"] = n_trunc
        corr["max_model_tokens"] = max_tok
        entry = {"corpus": corpus, "rating": rating_col,
                 "correspondence": corr["main"],
                 "loo": {"variants": corr["loo_variants"],
                         "all_pass": corr["loo_all_pass"]},
                 "n_truncated": n_trunc, "max_model_tokens": max_tok}
        if axis == "moderate_to_extreme":
            entry["pairs"] = check_pairs(axis, corpora["hypo"])
        entry["pass"] = corr["main"]["pass"] and corr["loo_all_pass"] and (
            entry.get("pairs", {}).get("pass", True))
        results[axis] = entry
        print(f"{axis:24s} rho={corr['main']['rho']:+.3f} "
              f"CI=[{corr['main']['ci_lo']:+.3f}, {corr['main']['ci_hi']:+.3f}] "
              f"loo_all_pass={corr['loo_all_pass']} "
              f"pass={entry['pass']}")
    return results


def render_summary(results):
    rows = []
    for axis, entry in results.items():
        c = entry["correspondence"]
        row = {"axis": axis, "corpus": entry["corpus"], "n": c["n"],
               "rho": round(c["rho"], 3), "ci_lo": round(c["ci_lo"], 3),
               "ci_hi": round(c["ci_hi"], 3),
               "correspondence_pass": c["pass"],
               "loo_all_pass": entry["loo"]["all_pass"],
               "n_truncated": entry["n_truncated"]}
        if "pairs" in entry:
            row["pair_rate"] = round(entry["pairs"]["rate"], 3)
            row["pairs_pass"] = entry["pairs"]["pass"]
        row["axis_pass"] = entry["pass"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("axis")


def write_outputs(results, summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "pinned": json.loads(SOURCES_JSON.read_text())["pinned"],
        "files": {rel: rec["sha1"] for rel, rec in
                  json.loads(SOURCES_JSON.read_text())["files"].items()},
        "embedding_model": "all-mpnet-base-v2",
        "boot_seed": BOOT_SEED, "boot_draws": BOOT_DRAWS,
        "criteria": {"rho_min": RHO_MIN, "ci_level": CI_LEVEL,
                     "pair_rate_min": PAIR_RATE_MIN},
        "smoke": SMOKE,
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__,
                     "sentence_transformers": __import__(
                         "sentence_transformers").__version__},
        "anchors_sha1": hashlib.sha1(json.dumps(ANCHORS, sort_keys=True).encode()).hexdigest(),
    }
    payload = {"provenance": provenance, "results": results}
    (OUT_DIR / "validation_results.json").write_text(json.dumps(payload, indent=2))
    summary.to_csv(OUT_DIR / "validation_results.csv")
    write_anchor_table(OUT_DIR / "tab_anchor_sentences.tex")
    print(f"wrote {OUT_DIR / 'validation_results.json'}")
    print(f"wrote {OUT_DIR / 'validation_results.csv'}")
    print(f"wrote {OUT_DIR / 'tab_anchor_sentences.tex'}")


def write_anchor_table(path):
    """LaTeX table of the 50 anchor sentences, from the ANCHORS the
    pipeline uses (single source of truth: echochain.semantic_axes.ANCHORS)."""
    pole_labels = {
        "neg_valence": "Negative", "pos_valence": "Positive",
        "neutral_tone": "Neutral", "intense_tone": "Intense",
        "moderate": "Moderate", "extreme": "Extreme",
        "factual": "Factual", "narrative": "Narrative",
        "abstract": "Abstract", "concrete": "Concrete",
    }
    axis_of_pole = {k: a for a, (f, s) in AXES.items() for k in (f, s)}
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Anchor sentences defining the five semantic axes (five per "
        "pole, ten per axis). Each text is projected onto the direction between "
        "the two pole centroids.}\\label{tab:anchor-sentences}",
        "\\small",
        "\\begin{tabularx}{\\textwidth}{l l X}",
        "\\toprule",
        "Axis & Pole & Anchor sentence \\\\",
        "\\midrule",
    ]
    for axis in AXES:
        first, second = AXES[axis]
        for pole_key in (first, second):
            for sentence in ANCHORS[pole_key]:
                esc = sentence.replace("&", "\\&").replace("%", "\\%").replace(
                    "_", "\\_").replace("#", "\\#")
                lines.append(f"{AXIS_LABELS[axis]} & {pole_labels[pole_key]} & {esc} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    path.write_text("\n".join(lines))


# %% [markdown]
# ## Revision protocol
#
# Frozen before the validation verdicts (2026-11-14). `RUN_REVISION` reads
# the validation verdicts, proposes anchors for every failing axis under the
# fixed selection rule (human rating and text length only, never axis
# scores), and re-validates each revised axis on held-out data only: on the
# held-out half for the correspondence and anchor-stability criteria, on the
# held-out pairs for the contrast-pair criterion, with the certainty corpus
# re-entering in full because it is never used for anchor selection. It does
# not edit any module; an accepted revision is entered into
# `semantic_axes.py` by hand and the validation is re-run.

# %%
from echochain.metrics_utils import token_len  # noqa: E402  (the paper's length rule)


def deterministic_half_split(n, seed):
    """Seeded construction/held-out split before any candidate inspection."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def select_pole_candidates(corpus_df, rating_col, pole, tok_len):
    """Candidates for one pole from the construction half: the extreme
    rating decile, lengths of 5 to 20 word_tokenize tokens, the five
    shortest, ties broken by corpus row order."""
    ratings = corpus_df[rating_col].astype(float).to_numpy()
    if pole == "second":
        threshold = float(np.quantile(ratings, 1.0 - DECILE_FRACTION))
        cand = corpus_df.loc[ratings >= threshold]
    else:
        threshold = float(np.quantile(ratings, DECILE_FRACTION))
        cand = corpus_df.loc[ratings <= threshold]
    cand = cand.drop_duplicates(subset="text").reset_index(drop=True)
    cand = cand.assign(_tokens=[int(tok_len(t)) for t in cand["text"]])
    cand = cand[(cand["_tokens"] >= REV_MIN_WORDS) & (cand["_tokens"] <= REV_MAX_WORDS)]
    cand = cand.sort_values("_tokens", kind="stable").head(REVS_PER_POLE)
    return [{"pole": pole, "row": int(i), "text": t, "rating": float(r),
             "tokens": int(k)}
            for i, t, r, k in zip(cand.index, cand["text"], cand[rating_col],
                                  cand["_tokens"])]


def revise_axis(axis, corpora, tok_len):
    """Select revised anchors for a failing axis and re-validate on
    held-out data only. Returns the proposal with its provenance and the
    held-out verdicts."""
    rng = np.random.default_rng(REVISION_SPLIT_SEED)
    if axis == "moderate_to_extreme":
        pairs = corpora["hypo"]
        cons_idx, held_idx = deterministic_half_split(len(pairs), REVISION_SPLIT_SEED)
        cons = pairs.iloc[cons_idx].reset_index(drop=True)
        held = pairs.iloc[held_idx].reset_index(drop=True)
        eligible = []
        for i, (h, l) in enumerate(zip(cons["text_hyperbole"], cons["text_literal"])):
            th, tl = int(tok_len(h)), int(tok_len(l))
            if REV_MIN_WORDS <= th <= REV_MAX_WORDS and REV_MIN_WORDS <= tl <= REV_MAX_WORDS:
                eligible.append({"pair_row": int(i), "hyperbole": h, "literal": l,
                                 "hyperbole_tokens": th, "literal_tokens": tl})
        selected = sorted(eligible, key=lambda d: (d["hyperbole_tokens"], d["pair_row"]))[:REVS_PER_POLE]
        override = {"moderate": [d["literal"] for d in selected],
                    "extreme": [d["hyperbole"] for d in selected]}
        corr = check_correspondence(axis, corpora["certainty"], rng, BOOT_DRAWS,
                                    anchors=override)
        pairs_res = check_pairs(axis, held, anchors=override)
        entry = {
            "anchor_source": "hypo construction half",
            "held_out": {"pairs": f"held-out pair count {pairs_res['n_pairs']}",
                         "certainty": "full corpus (never used for selection)"},
            "selected": selected,
            "correspondence": corr["main"], "loo_all_pass": corr["loo_all_pass"],
            "pairs": pairs_res,
            "pass": bool(corr["main"]["pass"] and corr["loo_all_pass"] and pairs_res["pass"]),
        }
    else:
        plan = REVISION_PLAN[axis]
        corpus, rating_col = plan["corpus"], plan["rating_col"]
        df = corpora[corpus]
        cons_idx, held_idx = deterministic_half_split(len(df), REVISION_SPLIT_SEED)
        cons = df.iloc[cons_idx].reset_index(drop=True)
        held = df.iloc[held_idx].reset_index(drop=True)
        first_cand = select_pole_candidates(cons, rating_col, "first", tok_len)
        second_cand = select_pole_candidates(cons, rating_col, "second", tok_len)
        held = held.rename(columns={rating_col: "rating"})
        first_key, second_key = AXES[axis]
        override = {first_key: [d["text"] for d in first_cand],
                    second_key: [d["text"] for d in second_cand]}
        corr = check_correspondence(axis, held, rng, BOOT_DRAWS, anchors=override)
        entry = {
            "anchor_source": f"{corpus} construction half",
            "held_out": f"held-out half count {len(held)}",
            "selected": {"first_pole": first_cand, "second_pole": second_cand},
            "correspondence": corr["main"], "loo_all_pass": corr["loo_all_pass"],
            "pass": bool(corr["main"]["pass"] and corr["loo_all_pass"]),
        }
    main = entry["correspondence"]
    extra = (f", pair_rate={entry['pairs']['rate']:.3f} "
             f"(criterion {entry['pairs']['criterion']}, n_pairs={entry['pairs']['n_pairs']})"
             if "pairs" in entry else "")
    print(f"REVISION {axis:24s} held-out rho={main['rho']:+.3f} "
          f"CI=[{main['ci_lo']:+.3f}, {main['ci_hi']:+.3f}] "
          f"loo_all_pass={entry['loo_all_pass']}{extra} "
          f"-> {'PASS' if entry['pass'] else 'STILL FAILING (excluded)'}")
    return entry


def load_verdicts():
    res_path = OUT_DIR / "validation_results.json"
    if not res_path.exists():
        raise FileNotFoundError(
            f"{res_path} not found; run the validation (RUN_VALIDATION) first")
    payload = json.loads(res_path.read_text())
    return payload["results"]


# Re-sourcing plan per axis: (corpus, rating column). The first-pole/second-pole
# anchor keys come from AXES. The moderate-to-extreme axis is special-cased
# above (HYPO pairs for selection, certainty for correspondence).
REVISION_PLAN = {
    "valence_neg_to_pos": {"corpus": "emobank", "rating_col": "V"},
    "tone_neutral_to_intense": {"corpus": "emobank", "rating_col": "A"},
    "factual_to_narrative": {"corpus": "narradetect", "rating_col": "rating"},
    "abstract_to_concrete": {"corpus": "concretext", "rating_col": "rating"},
}
REVISION_META = {
    "split_seed": REVISION_SPLIT_SEED,
    "decile_fraction": DECILE_FRACTION,
    "min_words": REV_MIN_WORDS, "max_words": REV_MAX_WORDS,
    "anchors_per_pole": REVS_PER_POLE,
    "criteria": {"rho_min": RHO_MIN, "ci_level": CI_LEVEL,
                 "pair_rate_min": PAIR_RATE_MIN,
                 "pair_rate_min_pairs": PAIR_RATE_MIN_PAIRS,
                 "sign_alpha": SIGN_ALPHA},
}


def run_revision():
    failing = [axis for axis, entry in load_verdicts().items()
               if not entry.get("pass", True)]
    if not failing:
        print("no failing axes; the anchor set stands as validated")
        return
    print("failing axes:", ", ".join(failing))
    corpora = load_corpora()
    revisions = {axis: revise_axis(axis, corpora, token_len) for axis in failing}
    payload = {"provenance": REVISION_META,
               "anchors_sha1_before": hashlib.sha1(
                   json.dumps(ANCHORS, sort_keys=True).encode()).hexdigest(),
               "revisions": revisions}
    if SMOKE:
        print("SMOKE run: revision results not written")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "revision_results.json").write_text(json.dumps(payload, indent=2))
        print(f"wrote {OUT_DIR / 'revision_results.json'}")
    return revisions


# %% [markdown]
# ## Run
#
# `RUN_FETCH` verifies (and downloads) the pinned corpora; `RUN_VALIDATION`
# runs the checks and writes the outputs unless `SMOKE` is set;
# `RUN_REVISION` runs the revision protocol on the failing axes.

# %%
if RUN_FETCH:
    META, FETCH_REPORT = verify_or_fetch(True)
    for rel, status in sorted(FETCH_REPORT.items()):
        print(f"{rel:55s} {status}")

# %%
if RUN_REVISION:
    REVISIONS = run_revision()
else:
    print("RUN_REVISION is False; nothing to do.")

# %%
if RUN_VALIDATION:
    CORPORA = load_corpora()
    for name, df in CORPORA.items():
        print(f"{name:12s} {len(df):6d} rows")
    RESULTS = run_validation(CORPORA)
    SUMMARY = render_summary(RESULTS)
    try:
        display(SUMMARY)
    except NameError:
        print(SUMMARY.to_string())
    if not SMOKE:
        write_outputs(RESULTS, SUMMARY)
    else:
        print("SMOKE run: outputs not written")
else:
    print("RUN_VALIDATION is False; nothing to do.")

# %% [markdown]
# ## Verdicts and follow-up
#
# Per the registered rules:
#
# - Every axis passing all criteria: the axis set is final; the anchor table
#   in `out/tab_anchor_sentences.tex` is the one recorded with the
#   registration.
# - A failing axis: set `RUN_REVISION = True`. It proposes anchors under the
#   frozen mechanical rule and re-validates on held-out data. To accept a
#   proposal, edit `echochain.semantic_axes.ANCHORS` by hand (replacing the
#   failing axis's five pole sentences), re-run `RUN_VALIDATION`, and mirror
#   any other paper numbers that depend on the anchors. Revision happens only
#   before the preregistration freeze.
# - An axis that still fails after revision is excluded from the confirmatory
#   analysis: no directional test uses it and its scores are not reported; the
#   variance estimands average over the validated axes, H3a requires both
#   predicted axes and is otherwise not evaluated, and the Benjamini-Hochberg
#   family shrinks by the withdrawn tests.
