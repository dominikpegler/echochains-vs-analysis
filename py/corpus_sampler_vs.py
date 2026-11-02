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
# Those source extractors still need to build the frozen manifests.
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
# python py/corpus_sampler_vs.py #   --manifest /path/to/frozen_candidates.jsonl
# ```
#
# The default output directory is `$DATA_DIR/data/corpus/sampler`; pass
# `--outdir` to change it.
#
# For a small interface test, use lower quotas:
#
# ```bash
# python py/corpus_sampler_vs.py #   --manifest /path/to/test_candidates.jsonl #   --outdir /tmp/opencode/corpus_sampler_test #   --quota 2 #   --reserve 1
# ```
#
# From inside the notebook, call the same entry point explicitly:
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
# script writes the final seeds, reserves, screen failures, draw rejections,
# quota tables, source audits, label audits, and a sampler manifest to the
# output directory.

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


def load_candidates(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for i, row in enumerate(read_manifest(path)):
            rows.append(normalize_row(row, str(path), i))
    if not rows:
        raise ValueError("no candidate rows loaded")
    return pd.DataFrame(rows)

# %% [markdown]
# ## Seed screen
#
# The screen is applied before model collection and never uses model
# outputs. Rejections are logged with a reason. Replacements are drawn only
# from the reserve list under the same quota.

# %%
def screen_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("_missing"):
        reasons.append("missing_required_metadata")
    if not row.get("text"):
        reasons.append("empty_text")
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


def apply_screen(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["screen_reason"] = out.apply(screen_reason, axis=1)
    out["screen_pass"] = out["screen_reason"].eq("")
    # Duplicate text hashes fail after the first randomly ordered row later;
    # here we only mark the duplicate group size for auditing.
    out["duplicate_hash_count"] = out.groupby("text_hash")["text_hash"].transform("size")
    return out

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


def draw_corpus(df: pd.DataFrame, cfg: DrawConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(cfg.seed)
    global_hashes: set[str] = set()
    selected_idx: list[int] = []
    reserve_idx: list[int] = []
    reject_rows: list[dict[str, Any]] = []
    for band in ("short", "long"):
        for genre in GENRES:
            sel, res, rej = draw_cell(df, band, genre, cfg, rng, global_hashes)
            selected_idx.extend(sel)
            reserve_idx.extend(res)
            reject_rows.extend(rej)
    selected = df.loc[selected_idx].copy().reset_index(drop=True)
    reserves = df.loc[reserve_idx].copy().reset_index(drop=True)
    selected["set"] = selected["band"].map({"short": "short_text", "long": "long_text"})
    order = selected.groupby(["band", "genre"]).cumcount() + 1
    selected["seed_id"] = [seed_id(b, g, int(n))
                            for b, g, n in zip(selected["band"],
                                               selected["genre"], order - 1)]
    reserves["reserve_id"] = [f"RES_{i+1:04d}" for i in range(len(reserves))]
    return selected, reserves, pd.DataFrame(reject_rows)


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
                  screened: pd.DataFrame, outdir: Path, cfg: DrawConfig) -> None:
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
    screened.loc[~screened["screen_pass"]].to_csv(outdir / "screen_failures.csv", index=False)
    meta = {
        "config": asdict(cfg),
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
    raw = load_candidates(args.manifest)
    screened = apply_screen(raw)
    selected, reserves, rejects = draw_corpus(screened, cfg)
    write_outputs(selected, reserves, rejects, screened, outdir, cfg)
    print("selected")
    print(quota_table(selected).to_string())
    print(f"wrote {outdir}")


if __name__ == "__main__" and "__file__" in globals():
    main()
