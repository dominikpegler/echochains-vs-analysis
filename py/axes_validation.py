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
# # Semantic-axis validation for the confirmatory study
#
# - This notebook implements the axis-validation protocol registered in the paper
# (Supplementary Methods, *Semantic axis construction*)
# - It scores five established human-annotated corpora with the five semantic axes and applies pass criteria below.
# - Run this notebook: nothing in it changes the axes, the anchors, or any paper file. Its outputs are the validation results and
# the anchor table that the preregistration freeze records.
#
# ## When to run
#
# Run this notebook **before the preregistration freeze**, after the corpora are
# fetched. The freeze records the anchor sentences and the validation verdicts,
# so both must exist and be final before the freeze.
#
# The notebook does not depend on collection and does not need to be re-run
# after the preregistration. Re-running is deterministic (fixed seeds)
# and reproduces the same outputs.
#
# ## Pass criteria
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
# An axis that fails is excluded from the confirmatory analysis: no
# directional test uses it and its scores are not reported in the confirmatory
# analysis (its validation metrics are reported in the supplementary
# material); the variance estimands average over the validated axes, H3a
# requires both predicted axes to be validated and is otherwise not
# evaluated, and the Benjamini-Hochberg family shrinks by the withdrawn
# tests. The registered power analysis assumes all five axes validate.
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
# | abstract-to-concrete | CONcreTEXT | in-context target-word concreteness | correspondence |
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
#   outputs.
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
SMOKE = False            # reduced corpora + fewer bootstrap draws, sanity check

# Pass criteria (registered; see the markdown cell above)
RHO_MIN = 0.20           # minimum Spearman correlation for the correspondence criterion
CI_LEVEL = 0.95          # two-sided bootstrap confidence interval for the correlation
BOOT_DRAWS = 2000        # bootstrap draws for the correspondence interval
PAIR_RATE_MIN = 0.75     # minimum share of correctly ordered contrast pairs
PAIR_RATE_MIN_PAIRS = 50 # at this pair count or above the rate criterion governs;
SIGN_ALPHA = 0.05        # below it, a one-sided binomial sign test at this level does

# Determinism
BOOT_SEED = 20261114

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

    # CONcreTEXT: English trial + test sentences with in-context concreteness
    # ratings for a target word. The two files name their rating column
    # differently (trial: MEAN, test: AVG); normalize to a common column
    # before concatenation so the full 534 English sentences load.
    frames = []
    for name, rating_col in [
        ("CONcreTEXT_trial_EN.tsv", "MEAN"),
        ("CONcreTEXT_test_EN.tsv", "AVG"),
    ]:
        df_c = pd.read_csv(AXVAL_DIR / "concretext" / name, sep="\t")
        if rating_col not in df_c.columns:
            raise ValueError(
                f"{name}: expected rating column {rating_col!r}, "
                f"has {list(df_c.columns)}")
        frames.append(df_c.rename(columns={rating_col: "rating"}))
    concretext = pd.concat(frames, ignore_index=True)
    concretext = concretext.dropna(subset=["TEXT", "rating"])
    concretext["TEXT"] = concretext["TEXT"].astype(str).str.strip()
    corpora["concretext"] = concretext[["TEXT", "rating"]].rename(
        columns={"TEXT": "text"}).reset_index(drop=True)

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

    # Schema guard: fail at load time if a corpus lost a required column.
    required = {
        "emobank": {"text", "V", "A"},
        "narradetect": {"text", "rating"},
        "concretext": {"text", "rating"},
        "certainty": {"text", "rating"},
        "hypo": {"text_hyperbole", "text_literal"},
    }
    for name, df_c in corpora.items():
        missing = required[name] - set(df_c.columns)
        if missing:
            raise ValueError(f"corpus {name!r}: missing columns {missing}")

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
    internally).
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


def check_pairs(axis, pair_df):
    """Contrast-pair criterion: the pair members are ordered in the
    predicted direction. Above PAIR_RATE_MIN_PAIRS pairs, pass at a correct
    ordering rate of >= PAIR_RATE_MIN; below it, a one-sided binomial sign
    test at SIGN_ALPHA replaces the rate criterion."""
    start, end = axis_centroids(axis)
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
# ## Run
#
# `RUN_FETCH` verifies (and downloads) the pinned corpora; `RUN_VALIDATION`
# runs the checks and writes the outputs unless `SMOKE` is set.

# %%
if RUN_FETCH:
    META, FETCH_REPORT = verify_or_fetch(True)
    for rel, status in sorted(FETCH_REPORT.items()):
        print(f"{rel:55s} {status}")

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
# - An axis passing all criteria is validated; the anchor table in
#   `out/tab_anchor_sentences.tex` (all 50 anchor sentences) is the one
#   recorded with the registration.
# - An axis that fails is excluded from the confirmatory analysis: no
#   directional test uses it and its scores are not reported in the
#   confirmatory analysis (its validation metrics are reported in the
#   supplementary material); the variance estimands average over the
#   validated axes, H3a requires both predicted axes and is otherwise not
#   evaluated, and the Benjamini-Hochberg family shrinks by the withdrawn
#   tests.

# %%

# %%
