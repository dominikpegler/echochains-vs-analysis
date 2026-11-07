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
#     version: 3.11.11
# ---

# %% [markdown]
# # Confirmatory corpus sampler for the verbalized-sampling study
#
# This notebook/script implements the seed-corpus sampling protocol from the
# paper. It is a **sampler**, not a source crawler. Running all cells in the
# notebook only loads the functions and checks that the notebook executes; it
# does not draw seeds unless you give it one or more frozen candidate
# manifests.
#
# ## Required input
#
# The input is one or more candidate manifests produced upstream by a source
# extraction/labeling workflow. Accepted formats are `.jsonl`, `.json`,
# `.csv`, and `.tsv`. Each row is one candidate text and must already carry
# source metadata, license information, offsets, and any automatic labels.
# This sampler does not fetch Wikipedia revisions, novels, or transcripts.
# The frozen manifest for the confirmatory study exists at
# `$DATA_DIR/data/corpus/vs_candidate_manifest.jsonl` (built 2026-11-07;
# 3,104,430 candidates; the article sample and its titles are recorded in
# the audit), so the draw can run as soon as the human label validation
# passes.
#
# Required columns for every row:
#
# ```text
# text, band, genre, source_id, source_type, source_title,
# source_date, license, offset, is_permissive, language
# ```
#
# Additional required fields by source type:
#
# - Wikipedia/encyclopedic rows need `revision_id`,
#   `revision_timestamp`, and `content_hash`.
# - Dialogue rows need `dialogue_turns >= 2` and `turn_ids`, because the
#   design uses contiguous multi-turn exchanges rather than isolated
#   utterances.
#
# Typical optional label fields are `topic`, `syntax_label`, `tense_label`,
# `temporal_reference`, and `auto_label_version`; the sampler preserves and
# audits them when present.
#
# ## How to run it
#
# From a terminal, pass at least one manifest:
#
# ```bash
# python py/corpus_sampler_vs.py --manifest /path/to/frozen_candidates.jsonl
# ```
#
# The default output directory is `$DATA_DIR/data/corpus/sampler`; pass
# `--outdir` to change it.
#
# For a small interface test, use lower quotas:
#
# ```bash
# python py/corpus_sampler_vs.py --manifest /path/to/test_candidates.jsonl \
#   --outdir /tmp/opencode/corpus_sampler_test --quota 2 --reserve 1
# ```
#
# Inside the notebook, the workflow cells at the end of this file run the
# same entry point with the frozen paths; a terminal gives the identical
# call:
#
# ```python
# main([
#     "--manifest", "/path/to/test_candidates.jsonl",
#     "--outdir", "/tmp/opencode/corpus_sampler_test",
#     "--quota", "2",
#     "--reserve", "1",
# ])
# ```
#
# ## Output
#
# The output is a coverage-balanced test corpus, not a representative sample
# of English. The final design has two sets: 90 short texts and 90 long
# texts. Each set has 30 encyclopedic, 30 novel, and 30 dialogue seeds. The
# genre label `novel` here is the pipeline's internal code; the paper calls
# this genre fiction. The script writes the final seeds, reserves, screen
# failures, draw rejections, quota tables, source audits, label audits, and
# a sampler manifest to the output directory.

# %%
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re

import nltk

try:  # one-time local setup for the length rule
    nltk.word_tokenize("A test sentence.")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# %% [markdown]
# ## Locked protocol constants

# %%
GENRES = ("encyclopedic", "novel", "dialogue")
BANDS = {
    "short": (10, 40),
    "long": (100, 200),
}
QUOTA_PER_CELL = 30
RESERVE_PER_CELL = 20
DEFAULT_SEED = 20261031

# Source caps apply within each band and genre. They prevent one article,
# book, film, or author from dominating a cell. The values are intentionally
# conservative; the draw aborts if a quota cannot be met.
DEFAULT_SOURCE_CAP = 3
DEFAULT_AUTHOR_CAP = 6

REQUIRED = {
    "text", "band", "genre", "source_id", "source_type", "source_title",
    "source_date", "license", "offset", "is_permissive", "language",
}
OPTIONAL = {
    "author", "revision_id", "revision_timestamp", "content_hash",
    "document_id", "topic", "syntax_label", "tense_label",
    "temporal_reference", "dialogue_turns", "turn_ids", "auto_label_version",
}
OUT_FIELDS = [
    "seed_id", "set", "band", "genre", "text", "word_count",
    "source_type", "source_id", "source_title", "author", "source_date",
    "license", "offset", "revision_id", "revision_timestamp",
    "document_id", "content_hash", "topic", "syntax_label",
    "tense_label", "temporal_reference", "dialogue_turns", "turn_ids",
    "auto_label_version", "text_hash",
]

# %% [markdown]
# ## Loading and normalization
#
# Candidate manifests can be `.jsonl`, `.json`, `.csv`, or `.tsv`. A row is
# a candidate seed. It must include text, band, genre, source metadata,
# language, license, and a permissive-license flag. Optional labels are
# preserved if present.

# %%
def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # accept {"candidates": [...]} or a dict-of-records
            if "candidates" in obj:
                obj = obj["candidates"]
            else:
                obj = list(obj.values())
        if not isinstance(obj, list):
            raise ValueError(f"JSON manifest must contain a list: {path}")
        return obj
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter=sep))
    raise ValueError(f"unsupported manifest suffix: {path}")


def coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def normalize_genre(x: str) -> str:
    y = str(x).strip().lower()
    aliases = {
        "wiki": "encyclopedic", "wikipedia": "encyclopedic",
        "encyclopedia": "encyclopedic", "encyclopedic": "encyclopedic",
        "fiction": "novel", "novel": "novel", "prose": "novel",
        "transcript": "dialogue", "transcripts": "dialogue",
        "dialogue": "dialogue", "dialog": "dialogue",
    }
    if y not in aliases:
        raise ValueError(f"unknown genre: {x!r}")
    return aliases[y]


def normalize_band(x: str) -> str:
    y = str(x).strip().lower()
    aliases = {"sentence": "short", "short": "short", "long": "long"}
    if y not in aliases:
        raise ValueError(f"unknown band: {x!r}")
    return aliases[y]


def collapse_ws(text: str) -> str:
    # Preserve line breaks between dialogue turns when present; collapse
    # within-line whitespace.
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines) if len(lines) > 1 else (lines[0] if lines else "")


def word_count(text: str) -> int:
    # Length rule frozen with the manifest builder: NLTK word_tokenize
    # token counts, matching the paper's delta-token/length metric
    # (echochains-core metrics_utils.token_len).
    return len(nltk.word_tokenize(text))


def text_hash(text: str) -> str:
    return hashlib.sha256(collapse_ws(text).encode("utf-8")).hexdigest()[:16]


def normalize_row(row: dict[str, Any], source_file: str, row_index: int) -> dict[str, Any]:
    missing = sorted(k for k in REQUIRED if k not in row or str(row.get(k, "")).strip() == "")
    out = dict(row)
    out["source_file"] = source_file
    out["row_index"] = row_index
    out["_missing"] = ";".join(missing)
    out["text"] = collapse_ws(out.get("text", ""))
    out["band"] = normalize_band(out.get("band", "")) if not missing else str(out.get("band", ""))
    out["genre"] = normalize_genre(out.get("genre", "")) if not missing else str(out.get("genre", ""))
    out["word_count"] = word_count(out["text"])
    out["is_permissive"] = coerce_bool(out.get("is_permissive"))
    out["language"] = str(out.get("language", "")).strip().lower()
    out["author"] = str(out.get("author", "")).strip()
    out["dialogue_turns"] = int(float(out.get("dialogue_turns", 0) or 0))
    out["text_hash"] = text_hash(out["text"])
    for field in OUT_FIELDS:
        out.setdefault(field, "")
    return out

# %% [markdown]
# ## Streaming load, screen, and draw for large manifests
#
# The frozen confirmatory manifest is 3.7 GB (3,104,430 rows). The
# full-load path (`pd.read_json` or `read_text` + DataFrame) exceeds RAM.
# The functions below stream the manifest in two passes, one per band
# ("short" first, then "long"): each pass normalizes rows one line at a
# time with identical order, indices, and normalization to the old
# full-load path, retains only its band's passing rows, and draws that
# band's cells in the canonical order with the shared rng and the
# cross-band duplicate set, so the selected, reserve, and rejection
# outputs match the full-load semantics exactly. Invalid-band rows are
# collected on the first pass, and the screen-failure audit is the union
# of both passes. The `.json`, `.csv`, and `.tsv` branches load in full
# (small sources) via `read_manifest`.

# %%
def stream_norm_rows(paths: list[Path]):
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            i = 0
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    yield normalize_row(json.loads(line), str(path), i)
                    i += 1
        else:
            for i, row in enumerate(read_manifest(path)):
                yield normalize_row(row, str(path), i)


def screen_and_group(paths: list[Path], phase: str,
                     hash_counter: Counter[str]) -> tuple[dict, pd.DataFrame, int]:
    """One streaming pass over the manifest for one phase ("short" or
    "long"). Rows of the phase's band are screened, grouped, and retained;
    the other phase's band rows are skipped except for two bookkeeping
    duties done on the first pass only (phase == "short"): counting
    `n_screened` and the full-manifest hash counter, so
    `duplicate_hash_count` keeps its whole-manifest semantics without being
    counted twice. Rows whose band is invalid are collected on the "short"
    pass, so the two passes partition the manifest and the screen-failure
    audit loses nothing. Retaining one band at a time keeps peak memory near
    that band's passing rows instead of the whole pool."""
    if phase not in BANDS:
        raise ValueError(f"phase must be a band: {sorted(BANDS)}")
    first = phase == "short"
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fails: list[dict[str, Any]] = []
    n_before = 0
    for row in stream_norm_rows(paths):
        if first:
            n_before += 1
            hash_counter[row["text_hash"]] += 1
        band = row["band"]
        if band != phase and not (phase == "short" and band not in BANDS):
            continue
        reason = screen_reason(row)
        row["screen_reason"] = reason
        row["screen_pass"] = reason == ""
        if row["screen_pass"] and row["band"] in BANDS and row["genre"] in GENRES:
            cells.setdefault((row["band"], row["genre"]), []).append(row)
        else:
            fails.append(row)
    for rows in cells.values():
        for row in rows:
            row["duplicate_hash_count"] = hash_counter[row["text_hash"]]
    for row in fails:
        row["duplicate_hash_count"] = hash_counter[row["text_hash"]]
    fails_df = pd.DataFrame(fails) if fails else pd.DataFrame()
    return cells, fails_df, n_before


def draw_corpus_stream(paths: list[Path], cfg: DrawConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Draw the confirmatory seed set from large manifests without loading
    them fully. Two streaming passes over the manifest, one per band
    (short first, then long), each retaining only that band's passing rows,
    so peak memory stays near one band instead of the whole pool; the draw
    runs the canonical per-cell draws in the fixed cell order with one
    shared rng and one cross-band `global_hashes` set, exactly as the
    full-load semantics prescribe. Returns
    (selected, reserves, rejects, screen_failures, n_screened)."""
    hash_counter: Counter[str] = Counter()
    rng = random.Random(cfg.seed)
    global_hashes: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    fail_frames: list[pd.DataFrame] = []
    n_screened = 0
    for phase in ("short", "long"):
        cells, fails_df, n_this = screen_and_group(paths, phase, hash_counter)
        if phase == "short":
            n_screened = n_this
        fail_frames.append(fails_df)
        for genre in GENRES:
            frame = pd.DataFrame(cells.get((phase, genre), []))
            sel, res, rej = draw_cell(frame, phase, genre, cfg, rng,
                                      global_hashes)
            selected_rows.extend(frame.loc[sel].to_dict(orient="records"))
            reserve_rows.extend(frame.loc[res].to_dict(orient="records"))
            reject_rows.extend(rej)
            del frame
            cells.pop((phase, genre), None)
    selected = pd.DataFrame(selected_rows)
    reserves = pd.DataFrame(reserve_rows)
    selected["set"] = selected["band"].map({"short": "short_text", "long": "long_text"})
    order = selected.groupby(["band", "genre"]).cumcount() + 1
    selected["seed_id"] = [seed_id(b, g, int(n))
                            for b, g, n in zip(selected["band"],
                                               selected["genre"], order - 1)]
    reserves["reserve_id"] = [f"RES_{i+1:04d}" for i in range(len(reserves))]
    fail_all = pd.concat(fail_frames, ignore_index=True) if any(
        len(f) for f in fail_frames) else pd.DataFrame()
    return selected, reserves, pd.DataFrame(reject_rows), fail_all, n_screened

# %% [markdown]
# ## Seed screen
#
# The screen is applied before model collection and never uses model
# outputs. Rejections are logged with a reason. Replacements are drawn only
# from the reserve list under the same quota.

# %%
ARTIFACT_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|\|thumb|\bFile:|\bImage:|\bCategory:|\bTemplate:")


def screen_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("_missing"):
        reasons.append("missing_required_metadata")
    if not row.get("text"):
        reasons.append("empty_text")
    if row.get("text") and ARTIFACT_RE.search(str(row.get("text"))):
        reasons.append("extraction_artifact")
    if row.get("language") not in {"en", "eng", "english"}:
        reasons.append("language_not_english")
    # The permissive-license promise covers the encyclopedic and fiction
    # cells. Dialogue sources (film transcripts) may be restricted-license
    # material, but their license string must be documented.
    if not row.get("is_permissive"):
        if row.get("genre") == "dialogue":
            if not str(row.get("license", "")).strip():
                reasons.append("license_not_documented")
        else:
            reasons.append("license_not_permissive")
    band = row.get("band")
    if band not in BANDS:
        reasons.append("bad_band")
    else:
        lo, hi = BANDS[band]
        if not (lo <= int(row.get("word_count", 0)) <= hi):
            reasons.append("length_out_of_band")
    genre = row.get("genre")
    if genre not in GENRES:
        reasons.append("bad_genre")
    if genre == "dialogue":
        if int(row.get("dialogue_turns", 0)) < 2:
            reasons.append("dialogue_not_multiturn")
        if not str(row.get("turn_ids", "")).strip():
            reasons.append("dialogue_missing_turn_ids")
    if "�" in str(row.get("text", "")):
        reasons.append("encoding_artifact")
    if str(row.get("source_type", "")).lower() == "wikipedia":
        if not str(row.get("revision_id", "")).strip():
            reasons.append("wiki_missing_revision_id")
        if not str(row.get("revision_timestamp", "")).strip():
            reasons.append("wiki_missing_revision_timestamp")
        if not str(row.get("content_hash", "")).strip():
            reasons.append("wiki_missing_content_hash")
    return ";".join(reasons)

# %% [markdown]
# ## Deterministic draw
#
# The draw is random within locked cells after screening. A source cap stops
# one source from dominating a cell. An author cap applies where author
# metadata exists. The selected rows and reserve rows are disjoint. If a
# quota cannot be filled, the script aborts and writes the audit tables so
# the source pool can be broadened.

# %%
@dataclass
class DrawConfig:
    seed: int = DEFAULT_SEED
    quota_per_cell: int = QUOTA_PER_CELL
    reserve_per_cell: int = RESERVE_PER_CELL
    source_cap: int = DEFAULT_SOURCE_CAP
    author_cap: int = DEFAULT_AUTHOR_CAP


def shuffled_indices(df: pd.DataFrame, rng: random.Random) -> list[int]:
    idx = list(df.index)
    rng.shuffle(idx)
    return idx


def draw_cell(df: pd.DataFrame, band: str, genre: str, cfg: DrawConfig,
              rng: random.Random, global_hashes: set[str]) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    cell = df[(df["screen_pass"]) & (df["band"] == band) & (df["genre"] == genre)]
    source_counts: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    selected: list[int] = []
    reserve: list[int] = []
    rejects: list[dict[str, Any]] = []
    target = cfg.quota_per_cell + cfg.reserve_per_cell
    for i in shuffled_indices(cell, rng):
        row = df.loc[i]
        sid = str(row["source_id"])
        author = str(row.get("author", ""))
        th = str(row["text_hash"])
        reasons = []
        if th in global_hashes:
            reasons.append("duplicate_text_hash_after_shuffle")
        if source_counts[sid] >= cfg.source_cap:
            reasons.append("source_cap")
        if author and author_counts[author] >= cfg.author_cap:
            reasons.append("author_cap")
        if reasons:
            rejects.append({"row_index": int(i), "band": band, "genre": genre,
                            "source_id": sid, "text_hash": th,
                            "reason": ";".join(reasons)})
            continue
        source_counts[sid] += 1
        if author:
            author_counts[author] += 1
        global_hashes.add(th)
        if len(selected) < cfg.quota_per_cell:
            selected.append(i)
        elif len(reserve) < cfg.reserve_per_cell:
            reserve.append(i)
        if len(selected) + len(reserve) >= target:
            break
    if len(selected) < cfg.quota_per_cell:
        raise RuntimeError(
            f"quota not filled for {band}/{genre}: selected {len(selected)} "
            f"of {cfg.quota_per_cell}; broaden the frozen source pool")
    if len(reserve) < cfg.reserve_per_cell:
        raise RuntimeError(
            f"reserve not filled for {band}/{genre}: reserve {len(reserve)} "
            f"of {cfg.reserve_per_cell}; broaden the frozen source pool")
    return selected, reserve, rejects


def seed_id(band: str, genre: str, i: int) -> str:
    b = "S" if band == "short" else "L"
    g = {"encyclopedic": "ENC", "novel": "NOV", "dialogue": "DIA"}[genre]
    return f"VS_{b}_{g}_{i+1:03d}"

# %% [markdown]
# ## Audits

# %%
def quota_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.pivot_table(index="band", columns="genre", values="text_hash",
                           aggfunc="count", fill_value=0)
            .reindex(index=["short", "long"], columns=list(GENRES)))


def source_audit(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["band", "genre", "source_id", "source_title", "author"]
    return (df.groupby(cols, dropna=False).size().rename("n")
            .reset_index().sort_values(["band", "genre", "n"], ascending=[True, True, False]))


def label_audit(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    fields = ["topic", "syntax_label", "tense_label", "temporal_reference"]
    out: dict[str, dict[str, int]] = {}
    for f in fields:
        if f in df.columns:
            vals = df[f].fillna("").replace("", "<missing>")
            out[f] = {str(k): int(v) for k, v in vals.value_counts().sort_index().items()}
    return out


def write_outputs(selected: pd.DataFrame, reserves: pd.DataFrame, rejects: pd.DataFrame,
                  screen_failures: pd.DataFrame, outdir: Path, cfg: DrawConfig,
                  n_screened: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    selected_cols = [c for c in OUT_FIELDS if c in selected.columns]
    reserve_cols = ["reserve_id"] + [c for c in OUT_FIELDS if c in reserves.columns and c != "seed_id"]
    selected[selected_cols].to_json(outdir / "vs_confirmatory_seeds_final.jsonl",
                                    orient="records", lines=True, force_ascii=False)
    reserves[reserve_cols].to_json(outdir / "vs_confirmatory_seed_reserves.jsonl",
                                   orient="records", lines=True, force_ascii=False)
    quota_table(selected).to_csv(outdir / "quota_table.csv")
    source_audit(selected).to_csv(outdir / "source_audit.csv", index=False)
    rejects.to_csv(outdir / "draw_rejections.csv", index=False)
    screen_failures.to_csv(outdir / "screen_failures.csv", index=False)
    meta = {
        "config": asdict(cfg),
        "n_screened": int(n_screened),
        "n_screen_failures": int(len(screen_failures)),
        "n_selected": int(len(selected)),
        "n_reserve": int(len(reserves)),
        "quota_table": quota_table(selected).to_dict(),
        "label_audit": label_audit(selected),
        "required_fields": sorted(REQUIRED),
        "optional_fields": sorted(OPTIONAL),
    }
    (outdir / "sampler_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# %% [markdown]
# ## Command-line entry point

# %%
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", type=Path, required=True,
                    help="candidate manifest (.jsonl, .json, .csv, .tsv); repeatable")
    ap.add_argument("--outdir", type=Path,
                    default=None,
                    help="output directory (default: $DATA_DIR/data/corpus/sampler)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quota", type=int, default=QUOTA_PER_CELL)
    ap.add_argument("--reserve", type=int, default=RESERVE_PER_CELL)
    ap.add_argument("--source-cap", type=int, default=DEFAULT_SOURCE_CAP)
    ap.add_argument("--author-cap", type=int, default=DEFAULT_AUTHOR_CAP)
    args = ap.parse_args(argv)

    cfg = DrawConfig(seed=args.seed, quota_per_cell=args.quota,
                     reserve_per_cell=args.reserve, source_cap=args.source_cap,
                     author_cap=args.author_cap)
    outdir = args.outdir
    if outdir is None:
        base = os.environ.get("DATA_DIR", str(Path.home() / "Dropbox/data_export/echochains"))
        outdir = Path(base).expanduser() / "data" / "corpus" / "sampler"
    selected, reserves, rejects, screen_failures, n_screened = draw_corpus_stream(args.manifest, cfg)
    write_outputs(selected, reserves, rejects, screen_failures, outdir, cfg, n_screened)
    print("selected")
    print(quota_table(selected).to_string())
    print(f"wrote {outdir}")


if __name__ == "__main__" and "__file__" in globals():
    main()

# %% [markdown]
# ## Workflow: draw the confirmatory seed set
#
# The draw runs ONCE, after the human label validation has passed and the
# label rules are frozen (dropping or merging label categories would change
# the screen, so it must not happen after the draw). The `DRAW_*` gates keep
# the file safe when the .py twin runs as a script and the notebook executes
# out of order: nothing is drawn or written until a gate is flipped
# deliberately. The frozen manifest and the default output paths point at
# `$DATA_DIR/data/corpus/`.

# %%
import os

DATA_DIR = Path(os.environ.get(
    "DATA_DIR", str(Path.home() / "Dropbox" / "data_export" / "echochains")))
FROZEN_MANIFEST = DATA_DIR / "data" / "corpus" / "vs_candidate_manifest.jsonl"
DEFAULT_OUTDIR = DATA_DIR / "data" / "corpus" / "sampler"
DRAW = False  # set True to run the one confirmatory draw
INSPECT = False  # set True after a draw exists to render the audit tables

# %%
if DRAW:
    main(["--manifest", str(FROZEN_MANIFEST),
          "--outdir", str(DEFAULT_OUTDIR)])

# %% [markdown]
# ### Post-draw inspection
#
# The audit tables the protocol logs: quotas per band x genre, per-source
# contributions, screen failures, draw rejections, and a preview of the
# 180 seeds per band for the final human review.

# %%
# %%
seed_df = None
if INSPECT and DEFAULT_OUTDIR.exists():
    try:
        seed_df = pd.read_json(DEFAULT_OUTDIR / "vs_confirmatory_seeds_final.jsonl",
                               lines=True)
    except ValueError as exc:
        print(f"cannot read the final seeds: {exc}")
if seed_df is not None and len(seed_df):
    print("== quota table ==")
    print(pd.read_csv(DEFAULT_OUTDIR / "quota_table.csv").to_string(index=False))
    print("== source audit (top 15) ==")
    print(pd.read_csv(DEFAULT_OUTDIR / "source_audit.csv").head(15).to_string(index=False))
    print("== screen failures ==")
    fails_path = DEFAULT_OUTDIR / "screen_failures.csv"
    if fails_path.exists() and fails_path.stat().st_size:
        print(f"{len(pd.read_csv(fails_path))} failures")
    else:
        print("0 (empty file)")
    print(f"== draw rejections: {len(pd.read_csv(DEFAULT_OUTDIR / 'draw_rejections.csv'))} rows ==")
    for band in ("short", "long"):
        preview = seed_df[seed_df["band"] == band]
        print(f"== {band} seeds ({len(preview)}) ==")
        print(preview[["seed_id", "band", "genre", "word_count",
                       "source_id", "text"]].to_string(index=False))
