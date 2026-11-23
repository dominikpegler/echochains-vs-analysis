"""Confirmatory chain generator for the VS paper (2026-10-31 design lock).

Runs the locked three-condition design (Direct, uniform-list joint control,
verbalized sampling) x four tasks (paraphrase, improve, take inspiration,
expand+compress) with 90 seeds x 5 chains x 200 transmitted hops per
task-per-condition cell. DECODING (2026-10-31 decision): every model runs
at its provider-recommended settings, temperature and top-p only (top-k and
repetition penalties excluded). gpt-6-luna (Azure) exposes NO sampling
parameters (verified 2026-10-31: temperature != 1 is an unsupported value
and top_p is an unsupported parameter on chat.completions AND the Responses
API, api-versions 2024-12-01-preview and 2025-04-01-preview); it runs at
the vendor default. VENDOR_DECODE holds the pinned panel's values
(Llama-3.1-8B 0.6/0.9, Qwen3-32B non-thinking 0.7/0.8, gemma-4-31B-it
1.0/0.95; card unverified entries are marked VERIFY AT EXECUTION there).
--decoding vendor (the design default) resolves per model; --decoding
default sends no sampling arguments (recorded as model_default); --decoding
locked is the pilot-matching 0.8/0.5 pair for exploratory-style reruns.
The exploratory pilot script `scripts/legacy_exploratory/generate_chains_vs.py` stays untouched as the
exploratory record.

ONE call per hop stage. Direct sends its single-turn instruction and
transmits the response. The uniform-list control sends one joint call that
requests k distinct candidates in a list-only JSON (no probabilities) and
selects one uniformly. VS sends one joint call that ALSO requests verbalized
probabilities and selects proportionally to the normalized probabilities.
The composite expand+compress hop is two stages: the selected enriched
candidate feeds compression, and the selected compressed text transmits.

EDGE-CASE POLICY (frozen here for the prereg; logged, never silently fixed):
- The API and selection seed is derived deterministically from model + task +
  condition + stage + seed + chain + hop + replicate (salt SALT); every retry
  of a hop stage reuses the SAME seed. Retries still help because real
  endpoints are only quasi-deterministic.
- Malformed JSON, a missing list key, or an unusable candidate list:
  retryable (up to --max-retries, default 3), each attempt counted.
- Unclosed final list object (the model writes the last candidate, then closes
  the array without closing the object): one missing `}` immediately before
  the array's closing bracket is repaired, and the hop logs the deviation
  "repaired_unclosed_object". The repair is logged, never silent, and is the
  only structural repair: unescaped quotes, missing commas, missing outer
  braces and every other break remain malformed. The raw response is cached
  unchanged, so a replay applies the same repair.
- Wrong candidate count: retryable; after the retries are exhausted a list
  with k' >= 2 candidates is ACCEPTED with the deviation
  "accepted_wrong_count" logged. With k' < 2 the stage FAILS.
- API errors (HTTP, timeouts): retryable within the same budget.
- Duplicate candidate texts: NOT retryable; logged under "duplicates".
- Missing or invalid probability fields (VS): NOT retryable; the probability
  is treated as 0.0 ("invalid_probability"); an all-zero list falls back to
  uniform weights ("uniform_fallback").
- Ties in selection weights: broken deterministically by the seeded RNG
  ("selection_tie").
- Retries exhausted (or k' < 2) or an empty input text: the stage failure is
  TERMINAL FOR THAT CHAIN ONLY. The chain is recorded as skipped
  (FAILED_MSG_<seed>_chain_<chain>.json: failing hop and stage, attempt
  count, error, the raw responses of every failed attempt, and the hops
  recorded so far) and the run continues with the next chain. That record is
  also the resume guard: a plain resume never re-runs a skipped chain. There
  is NO fallback to another condition.
- CELL-LEVEL POLICY (2026-11-21: the earlier pause guards were removed): a
  failed chain never stops the cell. Every (seed, chain) in the work list is
  attempted; a failure is recorded and skipped, and the run continues. There
  are no consecutive-failure or failure-share thresholds, and the retry-share
  pause was removed too, because a cell must always attempt the rest of its
  chains. Failed chains stay recoverable through the quality-control rerun
  (--rerun-failed: new replicate, fresh seeds).

RESUME / CACHE:
- A response-level cache (append-only JSONL) keyed on the SHA-256 of backend
  + model + message contents + decoding parameters + API seed. Only responses
  that PARSED cleanly are cached, so a malformed response is always re-drawn
  on resume. A crash resumes a chain from cached responses without
  re-calling the API; selection is reproducible because the RNG is seeded
  from make_seed. The cache append holds a lock, so `--workers > 1` can share
  one cache file across threads.
- Chain logs (per seed x chain JSON files) are written when the chain
  completes, mirroring the pilot's layout, and a partial log is flushed every
  --partial-flush-every hops so an unclean crash loses at most that many
  hops. Completed chains are skipped by their log file; chains recorded in
  FAILED_MSG_*.json are skipped as well.

CONCURRENCY (--workers N, default 1 = serial):
- The independent unit is one chain (seed x chain): hops are sequential
  within a chain, chains are not. N workers pull (seed, chain) units from one
  seed-major queue in a single process (threads; the work is HTTP-bound), so
  seeds still finish in order and the progress line stays readable.
- The set of calls does not depend on N: every API and selection seed derives
  from model + task + condition + stage + seed + chain + hop + replicate
  (make_seed), never from execution order. A serial and a parallel run of the
  same cell therefore select the same candidates; only the served sampling
  stream can differ, as it always can at a stochastic endpoint.
- Cells stay sequential (model x task x condition); `--workers` applies
  inside a cell. Choose N from the serving runtime's KV headroom: 16-32 on a
  single consumer GPU, 64 on an A100 node. Beyond roughly 64 in-flight
  requests a local 8B model is memory-bandwidth bound and more requests only
  add latency.
- A stage failure skips one chain; the cell always attempts its full work
  list, so nothing stops the run short of a crash or Ctrl-C.
- Output is compact by default: one progress line every 5% of the cell
  (chains processed, complete, skipped, elapsed, ETA), one line per skipped
  chain, and the final summary. --verbose adds a line per partial flush and
  per completed chain, for debugging a slow or stuck chain.

RERUNNING FAILURES (--rerun-failed):
- Reads the cell's FAILED_MSG_*.json records, takes the highest failed
  replicate r, and reruns exactly those (seed, chain) pairs at replicate r+1
  with fresh seeds from the new replicate index. The original logs and
  records are never overwritten. This is the registered quality-control
  rerun path (technical failure only, never substantive-output inspection).

STATUS REPORT (--status, read-only):
- Scans the logs root (one model, or every model directory found under it)
  and reports, per task x condition cell, how many chains are complete,
  failed (a FAILED_MSG_* record and no complete log; rerun with
  --rerun-failed), incomplete (a partial or truncated log; re-running the
  cell resumes it), and not started. Expected counts come from the drawn
  seed file per band, or from --n-seeds. The failed and incomplete chains are
  listed with their hop, stage, and error. Exit 0 regardless of gaps; a
  missing logs root exits 2. The notebook's last cell prints the same report
  for one model through study_status().

LOG SCHEMA (one JSON list per seed x chain; the hop-0 seed entry first):
  hop 0: {"hop": 0, "text": <seed text>, "task", "condition", "seed_id",
          "band", "genre", "model", "backend", "temperature", "top_p"}
  single-stage hop: {"hop", "task", "condition", "seed_id", "chain", "text"
          (transmitted), "random_seed", "model", "model_version",
          "temperature", "top_p", "raw_response", "candidates",
          "selected_index", "probabilities_normalized" (VS only, null
          otherwise), "attempts", "deviations", "from_cache"}
  composite hop: as above plus "stages": ordered per-stage dicts, each with
          {"stage", "random_seed", "raw_response", "candidates",
           "selected_index", "probabilities_normalized", "text_out",
           "attempts", "deviations", "from_cache"}; the top-level fields
          carry the LAST stage (compress), whose text is the transmitted one.

SEED FILE SCHEMA (--seeds-file, default $DATA_DIR/data/corpus/sampler/vs_confirmatory_seeds_final.jsonl; validated on load):
  {"band": "sentence" | "long", "seeds": [{"seed_id": 1, "text": "...",
    "genre": "encyclopedic" | "fiction" | "dialogue"}, ...]}
  The confirmatory seed sets themselves are built under plan step 5; the
  fixtures in scripts/confirmatory/test_confirmatory.py use synthetic seeds. The band must
  match the task (expand+compress: long; the other three tasks: sentence).

OUTPUT LAYOUT (under DATA_DIR/data by default, overridable):
  vs_data/logs/<model>/confirmatory/<task>/<condition>/MSG_<seed>_chain_<chain>.json
  vs_data/logs/<model>/confirmatory/<task>/<condition>/FAILED_MSG_<seed>_chain_<chain>.json
      (skipped-chain record: failure detail, failed raw responses, hop log)
  echodrift_confirmatory_cache/<backend>-<model>/<task>-<condition>.jsonl

Backends: "azure" (Luna; env API_KEY / AZURE_ENDPOINT / API_VERSION as in
the pilot) and "openai_compatible" (panel models on the Vienna cluster; env
OPENAI_BASE_URL / OPENAI_API_KEY).

Verify before any collection: python scripts/confirmatory/test_confirmatory.py (fixture
tests, no API calls), then a small live run (--limit-seeds 2 --limit-chains 1
--limit-hops 3).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

import vs_task_prompts
import tasks_registry

MAX_SIGNED_64 = (1 << 63) - 1
SALT = b"echochains-confirmatory-v1"
TEMPERATURE = 0.8
TOP_P = 0.5

TASK_ORDER = ("paraphrase", "improve", "take_inspiration", "expand+compress")
CONDITION_ORDER = ("direct", "uniform_list", "vs_weighted")
TASK_BANDS = {
    "paraphrase": "sentence",
    "improve": "sentence",
    "take_inspiration": "sentence",
    "expand+compress": "long",
}
DECODING = {"locked": {"temperature": 0.8, "top_p": 0.5},
            "default": {}}  # no sampling parameters sent
# Vendor-recommended decoding per pinned panel model (2026-10-31 decision:
# every model runs at its provider's recommended settings; temperature and
# top-p ONLY, top-k and repetition penalties excluded everywhere; verified
# against the official sources except where noted).
#   gpt-6-luna: NO sampling parameters exist (verified 2026-10-31 on both
#               chat.completions and the Responses API).
#   Llama-3.1:  Meta's model card recommendation (0.6/0.9); verify the
#   Qwen3-32B:  VERIFIED 2026-10-31 (official card, non-thinking mode).
#   gemma-4-31B-it: VERIFIED 2026-11-02 (Gemma 4 model card: temperature
#               1.0, top_p 0.95; the card's top_k 64 is excluded by the
#               no-top-k rule). Thinking disabled (drop the <|think|>
#               token from the system prompt); non-thinking runs emit an
#               empty thought block. Apache 2.0, ungated.
VENDOR_DECODE = {
    "gpt-6-luna": {},
    "meta-llama/Llama-3.1-8B-Instruct": {"temperature": 0.6, "top_p": 0.9},
    "Qwen/Qwen3-32B": {"temperature": 0.7, "top_p": 0.8},
    "google/gemma-4-31B-it": {"temperature": 1.0, "top_p": 0.95},
}
VENDOR_DECODE_SOURCE = {
    "gpt-6-luna": "vendor constraint (no parameters), verified 2026-10-31",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta model card; VERIFY AT EXECUTION",
    "Qwen/Qwen3-32B": "Qwen model card non-thinking mode; verified 2026-10-31",
    "google/gemma-4-31B-it": "Google Gemma 4 card; verified 2026-11-02",
}


def resolve_decoding(model: str, mode: str | None = None,
                     temperature: float | None = None,
                     top_p: float | None = None) -> dict:
    """Decoding parameters for one model from CLI arguments.

    Precedence: explicit --temperature/--top-p overrides everything (both
    must be given together); --decoding vendor uses VENDOR_DECODE (models
    outside the pinned table are rejected unless explicit values are given);
    --decoding default sends no sampling parameters; --decoding locked is
    the pilot-matching 0.8/0.5 pair (kept for exploratory-style reruns;
    NOT the confirmatory design's decoding).
    """
    if (temperature is None) != (top_p is None):
        raise ValueError("--temperature and --top-p must be given together")
    if temperature is not None:
        if not (0.0 < temperature <= 2.0 and 0.0 < top_p <= 1.0):
            raise ValueError(f"out-of-range sampling values: "
                             f"temperature={temperature}, top_p={top_p}")
        return {"temperature": temperature, "top_p": top_p}
    mode = mode or "vendor"
    if mode == "vendor":
        if model not in VENDOR_DECODE:
            raise ValueError(f"model {model!r} is outside the pinned panel; "
                             f"give explicit --temperature/--top-p or add it "
                             f"to VENDOR_DECODE with its card source")
        return dict(VENDOR_DECODE[model])
    return dict(DECODING[mode])


# ---------------------------------------------------------------------------
# Seeding, cache, backends
# ---------------------------------------------------------------------------

def make_seed(model: str, task: str, condition: str, stage_index: int,
              seed_id: int, chain_id: int, hop_id: int,
              replicate: int = 0) -> int:
    """Deterministic API + selection seed for one call.

    The pilot keyed on (sentence_id, chain_id, hop_id, replicate) only; the
    confirmatory design crosses tasks/conditions/stages, so the key adds all
    of them plus the model to avoid seed reuse across cells.
    """
    key = (
        f"{model}:{task}:{condition}:{stage_index}:{seed_id}:{chain_id}:"
        f"{hop_id}:{replicate}"
    ).encode() + SALT
    digest = hashlib.sha256(key).digest()
    return struct.unpack(">Q", digest[:8])[0] & MAX_SIGNED_64


def cache_key(backend: str, model: str, messages: list[dict],
              decoding: dict, random_seed: int) -> str:
    payload = json.dumps(
        [backend, model, messages, dict(decoding), random_seed],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CallCache:
    """Append-only JSONL response cache (response-level resume).

    Thread-safe: `workers > 1` shares one cache across the chain workers, so
    the append holds a lock. Reads are plain dict lookups and only ever
    insert complete records under the same lock.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.entries[rec["key"]] = rec
        self.hits = 0
        self._lock = threading.Lock()

    def has(self, key: str) -> bool:
        if key in self.entries:
            self.hits += 1
            return True
        return False

    def put(self, key: str, record: dict) -> None:
        with self._lock:
            self.entries[key] = dict(record, key=key)
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(self.entries[key], ensure_ascii=False) + "\n")

    def get(self, key: str) -> dict:
        return self.entries[key]


class CallResult:
    text: str
    model_version: str
    usage: dict

    def __init__(self, text: str, model_version: str, usage: dict | None = None):
        self.text = text
        self.model_version = model_version
        self.usage = usage or {}


class AzureBackend:
    """Luna via Azure OpenAI (same environment variables as the pilot)."""

    name = "azure"

    def __init__(self):
        from openai import AzureOpenAI  # imported lazily; fixtures never need it
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.environ["API_KEY"],
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
        )

    def call(self, model: str, messages, decoding: dict,
             random_seed: int) -> CallResult:
        resp = self.client.chat.completions.create(
            model=model, messages=messages, seed=random_seed,
            **decoding,
        )
        return CallResult(
            text=resp.choices[0].message.content.strip(),
            model_version=resp.model,
            usage={
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            })


class CompatibleBackend:
    """OpenAI-compatible chat endpoint (vLLM & co. on the Vienna cluster)."""

    name = "openai_compatible"

    def __init__(self):
        from openai import OpenAI  # imported lazily; fixtures never need it
        load_dotenv()
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )

    def call(self, model: str, messages, decoding: dict,
             random_seed: int) -> CallResult:
        resp = self.client.chat.completions.create(
            model=model, messages=messages, seed=random_seed,
            **decoding,
        )
        return CallResult(
            text=resp.choices[0].message.content.strip(),
            model_version=resp.model,
            usage={
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            })


# ---------------------------------------------------------------------------
# Parsing, selection, edge-case policy
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Retryable: no JSON, a missing list key, or no usable candidate list."""


class WrongCountError(Exception):
    """Not fixable by retrying: fewer than 2 usable candidates remain."""


def _parse_object(text: str) -> dict | None:
    """Parse one candidate string into a dict, tolerating LLM JSON quirks.

    Two narrow tolerances: `strict=False` accepts raw control characters
    (unescaped newlines and tabs inside candidate texts), and
    `ast.literal_eval` accepts Python-literal dicts (single quotes, trailing
    commas). A parse that is not a dict is rejected here, because every
    caller requires the task's list key.
    """
    try:
        obj = json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError, RecursionError):
        obj = None
    if isinstance(obj, dict):
        return obj
    try:
        obj = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _json_candidates(raw: str) -> list[str]:
    """The strings tried as the JSON object, widest first."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    return candidates


def extract_json_object(raw: str) -> dict:
    for candidate in _json_candidates(raw):
        obj = _parse_object(candidate)
        if obj is not None:
            return obj
    raise ParseError("no parseable JSON object in response")


def close_missing_last_brace(text: str) -> str | None:
    """Close the final list object when the model omitted its closing brace.

    Only the observed shape is repaired: a closed JSON string as the last
    value, no `}` after the last `{` (the final object is still open), the
    array closing bracket next, and the enclosing object closed. The repair
    inserts exactly one `}`; the result must then parse and carry the task's
    list key before it is accepted, so a wrong guess cannot pass.
    """
    end = text.rfind("]")
    if end == -1:
        return None
    head = text[:end]
    if not head.rstrip().endswith('"'):
        return None
    if head.rfind("{") <= head.rfind("}"):
        return None
    if not text[end + 1:].strip().startswith("}"):
        return None
    return head.rstrip() + "\n  }\n" + text[end:]


def extract_json_object_or_repair(raw: str) -> tuple[dict, bool]:
    """Parse; on failure try the missing-brace repair.

    Returns (object, repaired). Only a repair that yields a dict is accepted,
    and the caller logs `repaired_unclosed_object`, so no repair is silent.
    """
    try:
        return extract_json_object(raw), False
    except ParseError:
        pass
    for candidate in _json_candidates(raw):
        repaired = close_missing_last_brace(candidate)
        if repaired is None:
            continue
        obj = _parse_object(repaired)
        if obj is not None:
            return obj, True
    raise ParseError("no parseable JSON object in response")


def parse_joint_response(raw: str, key: str, k: int,
                         require_probabilities: bool) -> tuple[list[dict], list[str]]:
    """Parse one joint-list response into (candidates, deviations).

    candidates carry "text" and, for VS, "probability"; invalid probabilities
    become 0.0. Deviations follow the frozen edge-case policy in the module
    docstring. Raises ParseError for retryable breaks and WrongCountError
    when fewer than 2 usable candidates remain.
    """
    deviations: list[str] = []
    data, repaired = extract_json_object_or_repair(raw)
    if repaired:
        deviations.append("repaired_unclosed_object")
    if key not in data or not isinstance(data[key], list):
        raise ParseError(f"JSON missing the '{key}' list")
    items = data[key]
    if not all(isinstance(item, dict) for item in items):
        deviations.append("malformed_items")
        items = [item for item in items if isinstance(item, dict)]

    candidates: list[dict] = []
    for item in items:
        text = str(item.get("text", "") or "").strip()
        if not text:
            deviations.append("empty_text")
            continue
        cand = {"text": text}
        if require_probabilities:
            prob = item.get("probability", None)
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                prob = 0.0
            if prob <= 0.0:
                deviations.append("invalid_probability")
            cand["probability"] = max(0.0, prob)
        candidates.append(cand)

    if not candidates:
        raise ParseError("no usable candidate items in JSON list")
    if len(candidates) != k:
        deviations.append("wrong_count")
        if len(candidates) < 2:
            raise WrongCountError(f"{len(candidates)} usable candidate(s) < 2")
    texts = [c["text"] for c in candidates]
    if len(set(texts)) != len(texts):
        deviations.append("duplicates")
    return candidates, deviations


def normalize_probabilities(candidates: list[dict],
                            deviations: list[str]) -> list[float]:
    probs = [max(0.0, float(c.get("probability", 0.0))) for c in candidates]
    total = sum(probs)
    if total <= 0:
        deviations.append("uniform_fallback")
        return [1.0 / len(candidates)] * len(candidates)
    return [p / total for p in probs]


def select_index(candidates: list[dict], condition: str, rng: random.Random,
                 weights: list[float] | None, deviations: list[str]) -> int:
    """Selection index for one joint-list condition (seeded, reproducible)."""
    if condition == "uniform_list":
        return rng.randrange(len(candidates))
    if condition == "vs_weighted":
        assert weights is not None
        top = max(weights)
        if sum(1 for w in weights if w == top) > 1:
            deviations.append("selection_tie")
        return rng.choices(range(len(candidates)), weights=weights, k=1)[0]
    raise ValueError(f"selection is defined for joint-list conditions, "
                     f"got {condition!r}")


# ---------------------------------------------------------------------------
# Cell accounting and stage execution
# ---------------------------------------------------------------------------

class ChainStageFailure(Exception):
    """Terminal for one chain: no usable response within the retry budget.

    The chain is recorded (see ChainRunner._record_failure) and skipped; the
    run continues with the next chain. Nothing pauses the cell any more
    (2026-11-21 policy).
    """

    def __init__(self, hop_id: int, stage_name: str, attempts: int,
                 error: Exception | str,
                 failed_raws: list[str] | None = None):
        self.hop = hop_id
        self.stage = stage_name
        self.attempts = attempts
        self.error = error
        self.failed_raws = list(failed_raws or [])
        super().__init__(
            f"stage failed after {attempts} attempt(s) at hop {hop_id} "
            f"stage {stage_name}: {error}")


class CellState:
    """Per-cell accounting for progress reporting (no pausing).

    2026-11-21 policy: a failed chain only skips that chain. The cell never
    pauses, and every (seed, chain) in the work list is attempted. The
    counters below exist for the progress line and the summary; they cover
    chains finished in this process, so a resumed run starts from zero.

    With `workers > 1` the counters are shared under a lock.
    """

    def __init__(self, max_retries: int):
        self.max_retries = max_retries
        self.successes = 0
        self.retry_needed = 0
        self.chains_completed = 0
        self.chains_failed = 0
        self.consecutive_failures = 0
        self.lock = threading.Lock()

    def after_call(self, attempts: int) -> None:
        with self.lock:
            self.successes += 1
            if attempts > 1:
                self.retry_needed += 1

    def register_completion(self) -> None:
        with self.lock:
            self.chains_completed += 1
            self.consecutive_failures = 0

    def register_failure(self, failure: ChainStageFailure) -> None:
        """Count a skipped chain (the cell never pauses on failures)."""
        with self.lock:
            self.chains_failed += 1
            self.consecutive_failures += 1


def stage_messages(task: str, condition: str, stage_name: str,
                   stage_index: int, k: int, text_in: str) -> list[dict]:
    if condition == "direct":
        stage = tasks_registry.task_stages(task)[stage_index]
        return tasks_registry.build_messages(stage, text_in)
    if condition == "uniform_list":
        return vs_task_prompts.render_control(stage_name, text_in, k=k)
    if condition == "vs_weighted":
        return vs_task_prompts.render(stage_name, text_in, k=k)
    raise ValueError(f"unknown condition: {condition!r}")


def run_stage_call(backend, cache: CallCache, cell: CellState, *, model: str,
                   task: str, condition: str, stage_name: str,
                   stage_index: int, text_in: str, seed_id: int,
                   chain_id: int, hop_id: int, replicate: int = 0,
                   k: int = vs_task_prompts.VS_K,
                   decoding: dict | None = None) -> dict:
    """One API call for one hop stage, with cache, retries, and logging.

    Returns the stage's log record (see the module docstring). Raises
    ChainStageFailure when the retry budget is exhausted (the chain is
    terminal, the caller records and skips it). Nothing pauses the cell.
    """
    decoding = DECODING["locked"] if decoding is None else dict(decoding)
    if not text_in or not text_in.strip():
        raise ChainStageFailure(hop_id, stage_name, 0,
                                f"empty input text at seed {seed_id} "
                                f"chain {chain_id}")
    messages = stage_messages(task, condition, stage_name, stage_index, k,
                              text_in)
    random_seed = make_seed(model, task, condition, stage_index, seed_id,
                            chain_id, hop_id, replicate)
    ck = cache_key(backend.name, model, messages, decoding, random_seed)
    joint = condition != "direct"
    key = vs_task_prompts.VS_TEMPLATES[stage_name]["key"] if joint else None

    deviations: list[str] = []
    candidates: list[dict] | None = None
    selected_index: int | None = None
    probabilities: list[float] | None = None
    text_out: str | None = None
    model_version: str | None = None
    usage: dict = {}
    from_cache = False

    last_error: Exception | None = None
    accepted: dict | None = None
    failed_raws: list[str] = []
    for attempt in range(1, cell.max_retries + 1):
        if attempt == 1 and cache.has(ck):
            cached = cache.get(ck)
            raw = cached["text"]
            model_version = cached["model_version"]
            usage = dict(cached.get("usage"))
            from_cache = True
        else:
            try:
                fresh = backend.call(model, messages, decoding, random_seed)
            except Exception as exc:
                last_error = exc
                if attempt < cell.max_retries:
                    time.sleep(2.0 * attempt)
                continue
            raw = fresh.text
            model_version = fresh.model_version
            usage = fresh.usage
        if joint:
            try:
                candidates, parse_devs = parse_joint_response(
                    raw, key, k, require_probabilities=condition == "vs_weighted")
            except WrongCountError:
                # fewer than 2 usable candidates: retry first, then fail
                last_error = ParseError("fewer than 2 usable candidates")
                failed_raws.append(raw)
                continue
            except ParseError as exc:
                last_error = exc
                failed_raws.append(raw)
                continue
            if "wrong_count" in parse_devs and not from_cache \
                    and attempt < cell.max_retries:
                # a response with the exact count is preferred; retry before
                # accepting the mismatch (frozen policy above)
                last_error = ParseError("wrong candidate count")
                failed_raws.append(raw)
                continue
            if "wrong_count" in parse_devs:
                parse_devs.append("accepted_wrong_count")
            deviations.extend(parse_devs)
            if condition == "vs_weighted":
                probabilities = normalize_probabilities(candidates,
                                                        deviations)
            rng = random.Random(random_seed)
            selected_index = select_index(candidates, condition, rng,
                                          probabilities, deviations)
            text_out = candidates[selected_index]["text"]
        else:
            text_out = raw
        accepted = {
            "raw": raw, "model_version": model_version, "usage": usage,
            "candidates": candidates, "selected_index": selected_index,
            "probabilities_normalized": probabilities, "text_out": text_out,
            "attempts": attempt, "deviations": deviations,
            "from_cache": from_cache, "decoding": decoding,
        }
        # cache the response only once its parse is accepted, so resume never
        # reproduces a malformed or previously rejected response
        cache.put(ck, {"text": raw, "model_version": model_version,
                       "usage": usage})
        break
    if accepted is None:
        raise ChainStageFailure(hop_id, stage_name, cell.max_retries,
                                last_error, failed_raws)

    cell.after_call(accepted["attempts"])
    record = {
        "stage": stage_name,
        "random_seed": random_seed,
        "raw_response": accepted["raw"],
        "model_version": accepted["model_version"],
        "candidates": accepted["candidates"],
        "selected_index": accepted["selected_index"],
        "probabilities_normalized": accepted["probabilities_normalized"],
        "text_out": accepted["text_out"],
        "attempts": accepted["attempts"],
        "deviations": accepted["deviations"],
        "from_cache": accepted["from_cache"],
        "usage": accepted["usage"],
        "decoding": accepted["decoding"] or "model_default",
    }
    return record


def _fmt_duration(seconds: float) -> str:
    """Compact duration for progress lines: 45s, 18m, 2h36m."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class ChainRunner:
    """One task x condition cell: iterates seeds and chains, writes logs."""

    def __init__(self, *, backend, model: str, task: str, condition: str,
                 seeds: list[dict], band: str, out_dir: Path, cache: CallCache,
                 chains: int = 5, hops: int = 200, replicate: int = 0,
                 partial_flush_every: int = 25,
                 max_retries: int = 3,
                 vs_k: int = vs_task_prompts.VS_K,
                 decoding: dict | None = None,
                 verbose: bool = False):
        self.backend = backend
        self.model = model
        self.task = task
        self.condition = condition
        self.seeds = seeds
        self.band = band
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self.chains = chains
        self.hops = hops
        self.replicate = replicate
        self.partial_flush_every = partial_flush_every
        self.max_retries = max_retries
        self.vs_k = vs_k
        self.decoding = DECODING["locked"] if decoding is None \
            else dict(decoding)
        self.cell = CellState(max_retries=max_retries)
        self._stop = threading.Event()
        self._progress_lock = threading.Lock()
        self._progress_count = 0
        self._progress_total = 0
        self._progress_every = 1
        self._started = 0.0
        self.verbose = verbose

    # -- helpers ------------------------------------------------------------
    def _chain_path(self, seed_id: int, chain_id: int) -> Path:
        sid = f"{seed_id:03d}" if isinstance(seed_id, int) else str(seed_id)
        name = f"MSG_{sid}_chain_{chain_id:03d}"
        if self.replicate:
            name += f"_rep{self.replicate}"  # QC reruns never clobber the original
        return self.out_dir / f"{name}.json"

    @staticmethod
    def _log_complete(log: list[dict], hops: int) -> bool:
        return bool(log) and any(e.get("hop") == hops for e in log)

    def _failure_path(self, seed_id: int, chain_id: int) -> Path:
        sid = f"{seed_id:03d}" if isinstance(seed_id, int) else str(seed_id)
        name = f"FAILED_MSG_{sid}_chain_{chain_id:03d}"
        if self.replicate:
            name += f"_rep{self.replicate}"
        return self.out_dir / f"{name}.json"

    def _write(self, log: list[dict], path: Path, final: bool) -> None:
        target = path if final else path.with_suffix(".json.partial")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
        tmp.rename(target)

    def _record_failure(self, log: list[dict], seed_id: int, chain_id: int,
                        failure: ChainStageFailure) -> None:
        """Write the terminal record for one skipped chain.

        The file is the resume guard: a chain with a FAILED_MSG_* record is
        never re-run by a plain resume (only by a new --replicate or
        --rerun-failed). It also keeps the raw failed responses, which are
        not cached (only accepted parses are) and would otherwise be lost.
        """
        path = self._failure_path(seed_id, chain_id)
        record = {
            "status": "failed",
            "model": self.model,
            "backend": self.backend.name,
            "task": self.task,
            "condition": self.condition,
            "seed_id": seed_id,
            "chain": chain_id,
            "band": self.band,
            "genre": log[0].get("genre") if log else None,
            "replicate": self.replicate,
            "hops_recorded": max((e.get("hop", 0) for e in log), default=0),
            "failed_hop": failure.hop,
            "failed_stage": failure.stage,
            "attempts": failure.attempts,
            "error": str(failure.error),
            "error_type": type(failure.error).__name__,
            "failed_raw_responses": failure.failed_raws,
            "hop_log": log,
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
        tmp.rename(path)
        print(f"CHAIN SKIPPED: {path.name} (hop {failure.hop} "
              f"{failure.stage}: {failure.error})", flush=True)

    # -- chain loop -----------------------------------------------------------
    def run_chain(self, seed: dict, chain_id: int) -> str:
        """Run one chain; return "complete" or "failed" (recorded, skipped)."""
        seed_id = seed["seed_id"]
        path = self._chain_path(seed_id, chain_id)
        if self._failure_path(seed_id, chain_id).exists():
            # terminal skip: a plain resume never re-runs a failed chain
            return "failed"
        if path.exists():
            log = json.loads(path.read_text(encoding="utf-8"))
            if self._log_complete(log, self.hops):
                return "complete"
        else:
            log = [{
                "hop": 0, "text": seed["text"], "task": self.task,
                "condition": self.condition, "seed_id": seed_id,
                "band": self.band, "genre": seed.get("genre"),
                "model": self.model, "backend": self.backend.name,
                "temperature": self.decoding.get("temperature"),
                "top_p": self.decoding.get("top_p"),
            }]
        context = {"hop": 0, "stage": ""}
        started = time.monotonic()
        try:
            for hop_id in range(1, self.hops + 1):
                existing = next((e for e in log if e.get("hop") == hop_id), None)
                if existing is not None:
                    continue
                context["hop"] = hop_id
                self._run_hop(log, seed_id, chain_id, hop_id, context)
                if hop_id % self.partial_flush_every == 0 and hop_id < self.hops:
                    self._write(log, path, final=False)
                    if self.verbose:
                        print(f"  seed {seed_id} chain {chain_id}: "
                              f"hop {hop_id}/{self.hops}", flush=True)
        except ChainStageFailure as failure:
            # terminal for this chain only: record it and continue the cell
            self._record_failure(log, seed_id, chain_id, failure)
            self.cell.register_failure(failure)
            return "failed"
        self._write(log, path, final=True)
        partial = path.with_suffix(".json.partial")
        if partial.exists():
            partial.unlink()
        self.cell.register_completion()
        if self.verbose:
            print(f"  completed {seed_id} chain {chain_id} "
                  f"({self.hops} hops, {_fmt_duration(time.monotonic() - started)})",
                  flush=True)
        return "complete"

    def _run_hop(self, log: list[dict], seed_id: int, chain_id: int,
                 hop_id: int, context: dict) -> str:
        """Run one hop (both stages for the composite task); append entry."""
        composite = self.task == "expand+compress"
        stage_names = ["expand", "compress"] if composite else [self.task]
        if hop_id == 1:
            text_in = log[0]["text"]
        else:
            prev = next(e for e in log if e.get("hop") == hop_id - 1)
            text_in = prev["text"]
        stage_records: list[dict] = []
        stage_texts: list[str] = []
        for stage_index, stage_name in enumerate(stage_names):
            context["stage"] = stage_name
            record = run_stage_call(
                self.backend, self.cache, self.cell, model=self.model,
                task=self.task, condition=self.condition,
                stage_name=stage_name, stage_index=stage_index,
                text_in=text_in, seed_id=seed_id, chain_id=chain_id,
                hop_id=hop_id, replicate=self.replicate, k=self.vs_k,
                decoding=self.decoding)
            stage_records.append(record)
            if record["text_out"] is None:  # direct: the response transmits
                record["text_out"] = record["raw_response"]
            stage_texts.append(record["text_out"])
            text_in = record["text_out"]
            last = record
        top_probabilities = (
            last["probabilities_normalized"]
            if self.condition == "vs_weighted" else None)
        entry = {
            "hop": hop_id, "task": self.task, "condition": self.condition,
            "seed_id": seed_id, "chain": chain_id,
            "text": last["text_out"],
            "random_seed": last["random_seed"],
            "model": self.model, "model_version": last["model_version"],
            "temperature": self.decoding.get("temperature"),
            "top_p": self.decoding.get("top_p"),
            "raw_response": last["raw_response"],
            "candidates": last["candidates"],
            "selected_index": last["selected_index"],
            "probabilities_normalized": top_probabilities,
            "attempts": last["attempts"],
            "deviations": last["deviations"],
            "from_cache": last["from_cache"],
        }
        if composite:
            entry["stages"] = stage_records
            entry["stage_texts"] = stage_texts
        log.append(entry)
        return last["text_out"]

    # -- cell loop -------------------------------------------------------------
    def _note_progress(self) -> None:
        with self._progress_lock:
            self._progress_count += 1
            done = self._progress_count
        if done % self._progress_every != 0 and done != self._progress_total:
            return
        elapsed = time.monotonic() - self._started
        remaining = self._progress_total - done
        eta = elapsed / done * remaining if done else 0.0
        print(f"[{self.task}/{self.condition}] {done}/{self._progress_total} "
              f"chains ({100 * done / max(1, self._progress_total):.0f}%) | "
              f"{self.cell.chains_completed} complete, "
              f"{self.cell.chains_failed} skipped | "
              f"{_fmt_duration(elapsed)} elapsed, "
              f"~{_fmt_duration(eta)} left", flush=True)

    def _worker_task(self, seed: dict, chain_id: int) -> None:
        if self._stop.is_set():
            return
        self.run_chain(seed, chain_id)
        self._note_progress()

    def run(self, limit_seeds: int | None = None,
            limit_chains: int | None = None,
            limit_hops: int | None = None,
            workers: int = 1,
            only_pairs: list[tuple[str, int]] | None = None) -> None:
        """Run the cell. `workers > 1` runs chains concurrently.

        The work list is always (seed, chain) in seed-major order, so seeds
        finish in order and the set of calls is identical to a serial run:
        every API and selection seed derives from the design keys, not from
        execution order. A failed chain is skipped; the whole work list is
        always attempted.
        """
        if limit_hops is not None:
            self.hops = min(self.hops, limit_hops)
        chains = self.chains if limit_chains is None else min(self.chains,
                                                              limit_chains)
        seeds = self.seeds if limit_seeds is None else self.seeds[:limit_seeds]
        work = [(seed, chain_id) for seed in seeds
                for chain_id in range(1, chains + 1)]
        if only_pairs is not None:
            known = {str(seed["seed_id"]): seed for seed in seeds}
            missing = [pair for pair in only_pairs if pair[0] not in known]
            if missing:
                print(f"warning: {len(missing)} failed chain(s) not in the "
                      f"loaded seed set: {missing[:5]}"
                      f"{' ...' if len(missing) > 5 else ''}")
            work = [(known[str(seed_id)], chain_id)
                    for seed_id, chain_id in sorted(set(only_pairs))
                    if str(seed_id) in known]
        self._stop.clear()
        self._progress_count = 0
        self._progress_total = len(work)
        self._progress_every = max(1, self._progress_total // 20)
        self._started = time.monotonic()

        if workers <= 1:
            for seed, chain_id in work:
                if self._stop.is_set():
                    break
                self.run_chain(seed, chain_id)
                self._note_progress()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._worker_task, seed, chain_id)
                           for seed, chain_id in work]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    self._stop.set()
                    for future in futures:
                        future.cancel()
                    raise

        skipped = sorted(p.name for p in self.out_dir.glob("FAILED_MSG_*.json"))
        workers_note = f"; {workers} workers" if workers > 1 else ""
        print(f"cell complete: {self.task} / {self.condition} "
              f"({len(seeds)} seeds x {chains} chains x {self.hops} hops; "
              f"{self.cell.successes} successful calls, "
              f"{self.cell.retry_needed} needed a retry, "
              f"{self.cell.chains_completed} chains completed this run, "
              f"{self.cell.chains_failed} skipped this run{workers_note})")
        for name in skipped:
            print(f"  skipped chain record: {name}")
        if skipped and only_pairs is None:
            print("  rerun them with --rerun-failed (new replicate, "
                  "never overwrites the original)")
        elif skipped:
            print("  (records kept for provenance; the reruns write "
                  "separate _rep<n> files)")


# ---------------------------------------------------------------------------
# Seed loading and CLI
# ---------------------------------------------------------------------------

BANDS = ("sentence", "long")


# The sampler's draw writes one JSON record per line (JSONL) with the band
# per row ("short"/"long"); older hand-written files are a single JSON object
# {"band": ..., "seeds": [...]} with band "sentence"/"long". Accept both.
BAND_ALIASES = {"short": "sentence", "sentence": "sentence", "long": "long"}


def load_seed_file(path: Path, band: str | None = None) -> tuple[str, list[dict]]:
    """Load a seed file. The sampler's combined draw holds both bands in one
    JSONL file; pass `band` ("sentence"/"long") to select the task's set."""
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if band is not None:
            rows = [r for r in rows
                    if BAND_ALIASES.get(r.get("band"), r.get("band")) == band]
        bands = {BAND_ALIASES.get(r.get("band"), r.get("band")) for r in rows}
        if not bands:
            raise ValueError(f"{path}: no rows for band {band!r}")
        if len(bands) != 1:
            raise ValueError(f"{path}: mixed bands in JSONL seed file: {sorted(bands)}")
        band = bands.pop()
        seeds = rows
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        band = BAND_ALIASES.get(data.get("band"), data.get("band"))
        seeds = data.get("seeds")
    if band not in BANDS:
        raise ValueError(f"{path}: missing/unknown band; expected one of {BANDS}")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path}: missing a non-empty 'seeds' list")
    seen = set()
    for i, seed in enumerate(seeds):
        for required in ("seed_id", "text", "genre"):
            if required not in seed:
                raise ValueError(f"{path}: seed #{i} lacks '{required}'")
        if seed["seed_id"] in seen:
            raise ValueError(f"{path}: duplicate seed_id {seed['seed_id']}")
        seen.add(seed["seed_id"])
        if not isinstance(seed["text"], str) or not seed["text"].strip():
            raise ValueError(f"{path}: seed #{i} has an empty text")
    return band, seeds


def default_seeds_file() -> Path:
    return get_data_dir() / "data" / "corpus" / "sampler" / "vs_confirmatory_seeds_final.jsonl"


def get_data_dir() -> Path:
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise ValueError("environment variable DATA_DIR not set (.env missing?)")
    return Path(os.path.expanduser(data_dir))


def build_backend(name: str):
    if name == "azure":
        return AzureBackend()
    if name == "openai_compatible":
        return CompatibleBackend()
    raise ValueError(f"unknown backend: {name!r}")


def cell_out_dir(logs_root: Path, model: str, task: str, condition: str) -> Path:
    return logs_root / model / "confirmatory" / task / condition


def cache_file(cache_root: Path, backend: str, model: str, task: str,
               condition: str) -> Path:
    return (cache_root / f"{backend}-{model}" / f"{task}-{condition}.jsonl")


STATUS_NAME_RE = re.compile(
    r"^(?P<failed>FAILED_)?MSG_(?P<seed>.+)_chain_(?P<chain>\d+)"
    r"(?:_rep(?P<rep>\d+))?\.json(?P<partial>\.partial)?$")

STATUS_LIST_LIMIT = 50


def chain_statuses(cell_dir: Path, hops: int) -> dict[tuple[str, int], dict]:
    """Per (seed, chain) status for one cell, across every replicate present.

    Status is "complete" (a log reaches `hops`), "failed" (a FAILED_MSG_*
    record and no complete log), or "incomplete" (a partial or truncated log,
    resumable). Chains with no file are absent from the dict (never started).
    The highest complete replicate wins; a failure only counts when no
    replicate completed.
    """
    seen: dict[tuple[str, int], dict] = {}
    for path in cell_dir.glob("*MSG_*_chain_*.json*"):
        match = STATUS_NAME_RE.match(path.name)
        if match is None:  # e.g. a .json.tmp left mid-write
            continue
        key = (match["seed"], int(match["chain"]))
        entry = seen.setdefault(key, {"complete": [], "failed": [],
                                      "partial": [], "truncated": []})
        replicate = int(match["rep"] or 0)
        if match["failed"]:
            record = json.loads(path.read_text(encoding="utf-8"))
            entry["failed"].append({
                "replicate": replicate,
                "hop": record.get("failed_hop"),
                "stage": record.get("failed_stage"),
                "error": record.get("error"),
                "hops_recorded": record.get("hops_recorded"),
            })
            continue
        log = json.loads(path.read_text(encoding="utf-8"))
        last_hop = max((e.get("hop", 0) for e in log), default=0)
        bucket = ("partial" if match["partial"]
                  else "complete" if last_hop >= hops else "truncated")
        entry[bucket].append({"replicate": replicate, "last_hop": last_hop})
    statuses: dict[tuple[str, int], dict] = {}
    for key, entry in seen.items():
        if entry["complete"]:
            best = max(entry["complete"], key=lambda item: item["replicate"])
            statuses[key] = {"status": "complete", **best}
        elif entry["failed"]:
            worst = max(entry["failed"], key=lambda item: item["replicate"])
            statuses[key] = {"status": "failed", **worst}
        else:
            pool = entry["partial"] + entry["truncated"]
            best = max(pool, key=lambda item: item["last_hop"])
            statuses[key] = {"status": "incomplete", **best}
    return statuses


def study_status(logs_root: Path, *, model: str | None = None,
                 chains: int = 5, hops: int = 200,
                 seeds_by_band: dict[str, int | None] | None = None) -> dict:
    """Collect chain statuses for every task x condition cell.

    `model` limits the scan to one model directory; without it every
    `confirmatory` directory under `logs_root` is reported. `seeds_by_band`
    gives the expected number of seeds per band (from the drawn seed file);
    a missing or None entry falls back to the number of chains found on disk.
    """
    if model:
        models = [model]
    else:
        models = sorted(
            str(path.parent.relative_to(logs_root))
            for path in logs_root.rglob("confirmatory") if path.is_dir())
    report = {"logs_root": str(logs_root), "hops": hops, "models": []}
    for name in models:
        conf = logs_root / name / "confirmatory"
        cells = []
        for task in TASK_ORDER:
            for condition in CONDITION_ORDER:
                cell_dir = conf / task / condition
                statuses = chain_statuses(cell_dir, hops) \
                    if cell_dir.exists() else {}
                counts = {"complete": 0, "failed": 0, "incomplete": 0}
                for status in statuses.values():
                    counts[status["status"]] += 1
                expected = None
                if seeds_by_band is not None:
                    n_seeds = seeds_by_band.get(TASK_BANDS[task])
                    if n_seeds is not None:
                        expected = n_seeds * chains
                if expected is None:
                    expected = len(statuses)
                cells.append({
                    "task": task,
                    "condition": condition,
                    "expected": expected,
                    "seen": len(statuses),
                    "missing": max(0, expected - len(statuses)),
                    **counts,
                    "failed_chains": sorted(
                        (key, status) for key, status in statuses.items()
                        if status["status"] == "failed"),
                    "incomplete_chains": sorted(
                        (key, status) for key, status in statuses.items()
                        if status["status"] == "incomplete"),
                })
        report["models"].append({"model": name, "cells": cells})
    return report


def print_study_status(report: dict) -> None:
    """Print the report from study_status as a compact block."""
    print(f"logs root: {report['logs_root']}")
    hops = report.get("hops")
    for model_block in report["models"]:
        print(f"\nmodel: {model_block['model']}")
        cells = model_block["cells"]
        totals = {"complete": 0, "failed": 0, "incomplete": 0,
                  "missing": 0, "expected": 0}
        started, not_started = [], []
        for cell in cells:
            for name in ("complete", "failed", "incomplete", "missing",
                         "expected"):
                totals[name] += cell[name]
            (started if cell["seen"] else not_started).append(cell)
        for cell in started:
            extra = f"  missing {cell['missing']}" if cell["missing"] else ""
            print(f"  {cell['task']}/{cell['condition']:<13} "
                  f"{cell['complete']:>4}/{cell['expected']:<4}  "
                  f"failed {cell['failed']:<3} "
                  f"incomplete {cell['incomplete']:<3}{extra}")
        extra = f"  missing {totals['missing']:,}" if totals["missing"] else ""
        print(f"  {'total':<26} {totals['complete']:>4}/"
              f"{totals['expected']:<4}  failed {totals['failed']:<3} "
              f"incomplete {totals['incomplete']:<3}{extra}")
        if not_started:
            pending = sum(cell["missing"] for cell in not_started)
            names = ", ".join(f"{cell['task']}/{cell['condition']}"
                              for cell in not_started)
            print(f"  not started ({len(not_started)} cells, {pending:,} "
                  f"chains): {names}")
        failed = [(cell, key, status)
                  for cell in cells for key, status in cell["failed_chains"]]
        for cell, (seed, chain), status in failed[:STATUS_LIST_LIMIT]:
            print(f"  failed: {cell['task']}/{cell['condition']} {seed} "
                  f"chain {chain}: hop {status['hop']} {status['stage']}: "
                  f"{status['error']}")
        if len(failed) > STATUS_LIST_LIMIT:
            print(f"  failed: ... and {len(failed) - STATUS_LIST_LIMIT} more")
        incomplete = [(cell, key, status)
                      for cell in cells
                      for key, status in cell["incomplete_chains"]]
        for cell, (seed, chain), status in incomplete[:STATUS_LIST_LIMIT]:
            print(f"  incomplete: {cell['task']}/{cell['condition']} {seed} "
                  f"chain {chain}: last hop {status['last_hop']}"
                  + (f"/{hops}" if hops else ""))
        if len(incomplete) > STATUS_LIST_LIMIT:
            print(f"  incomplete: ... and "
                  f"{len(incomplete) - STATUS_LIST_LIMIT} more")


def failed_chains(out_dir: Path) -> tuple[int, list[tuple[str, int]]]:
    """Read the FAILED_MSG_* records of one cell.

    Returns (highest failed replicate, [(seed_id, chain), ...] at that
    replicate). `--rerun-failed` reruns exactly those pairs at replicate+1.
    The seed id is returned as a string (drawn seed ids are strings, e.g.
    "VS_S_ENC_001"; fixture seeds may be ints).
    """
    records: list[dict] = []
    for path in sorted(out_dir.glob("FAILED_MSG_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: unreadable failure record {path.name}: {exc}")
    if not records:
        return 0, []
    replicate = max(int(record.get("replicate", 0)) for record in records)
    pairs = sorted({(str(record["seed_id"]), int(record["chain"]))
                    for record in records
                    if int(record.get("replicate", 0)) == replicate})
    return replicate, pairs


def run_cell(backend, *, model: str, task: str, condition: str, band: str,
             seeds: list[dict], logs_root: Path, cache_root: Path,
             chains: int, hops: int, replicate: int, partial_flush_every: int,
             max_retries: int, limit_seeds: int | None,
             limit_chains: int | None, limit_hops: int | None,
             decoding: dict | None = None, workers: int = 1,
             rerun_failed: bool = False, verbose: bool = False) -> None:
    expected = TASK_BANDS[task]
    if band != expected:
        raise ValueError(f"seed band {band!r} does not match task {task!r}'s "
                         f"band {expected!r}")
    out_dir = cell_out_dir(logs_root, model, task, condition)
    only_pairs: list[tuple[str, int]] | None = None
    if rerun_failed:
        failed_rep, only_pairs = failed_chains(out_dir)
        if not only_pairs:
            print(f"no FAILED_MSG_* records in {out_dir}; nothing to rerun")
            return
        replicate = failed_rep + 1
        print(f"rerunning {len(only_pairs)} failed chain(s) from replicate "
              f"{failed_rep} as replicate {replicate}")
    cache = CallCache(cache_file(cache_root, backend.name, model, task,
                                 condition))
    runner = ChainRunner(
        backend=backend, model=model, task=task, condition=condition,
        seeds=seeds, band=band, out_dir=out_dir,
        cache=cache, chains=chains, hops=hops, replicate=replicate,
        partial_flush_every=partial_flush_every, max_retries=max_retries,
        decoding=decoding, verbose=verbose,
    )
    print(f"=== {task} / {condition} ===")
    runner.run(limit_seeds=limit_seeds, limit_chains=limit_chains,
               limit_hops=limit_hops, workers=workers, only_pairs=only_pairs)
    print(f"cache: {cache.hits} cached-response hits "
          f"({cache.path})")


def status_report(args) -> int:
    """Print the chain-status report for --status (read-only)."""
    load_dotenv()
    if args.logs_root:
        logs_root = Path(os.path.expanduser(args.logs_root))
    else:
        logs_root = get_data_dir() / "data" / "vs_data" / "logs"
    if not logs_root.exists():
        print(f"logs root does not exist: {logs_root}")
        return 2
    seeds_by_band: dict[str, int | None] | None
    if args.n_seeds is not None:
        seeds_by_band = {band: args.n_seeds for band in BANDS}
    else:
        try:
            seeds_path = (Path(args.seeds_file) if args.seeds_file
                          else default_seeds_file())
        except ValueError:
            print("note: DATA_DIR unset and no --seeds-file; expected counts "
                  "come from the chains found on disk")
            seeds_path = None
        seeds_by_band = None
        if seeds_path is not None:
            seeds_by_band = {}
            for band in BANDS:
                try:
                    _, band_seeds = load_seed_file(seeds_path, band=band)
                except (OSError, ValueError, KeyError,
                        json.JSONDecodeError) as exc:
                    print(f"note: no seed count for band {band!r} from "
                          f"{seeds_path} ({exc}); pass --n-seeds")
                    seeds_by_band[band] = None
                else:
                    seeds_by_band[band] = len(band_seeds)
    report = study_status(logs_root, model=args.model, chains=args.chains,
                          hops=args.hops, seeds_by_band=seeds_by_band)
    print_study_status(report)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirmatory chain generator (see the module docstring "
                    "for the design, edge-case policy, and log schema).")
    parser.add_argument("--task", choices=TASK_ORDER)
    parser.add_argument("--condition", choices=CONDITION_ORDER)
    parser.add_argument("--all-cells", action="store_true",
                        help="run all task x condition cells in order")
    parser.add_argument("--backend", choices=("azure", "openai_compatible"),
                        default="azure")
    parser.add_argument("--model",
                        help="model or deployment identifier (pinned in logs); "
                             "required for generation, optional with --status "
                             "(then every model under the logs root is reported)")
    parser.add_argument("--seeds-file", default=None,
                        help="seed JSON in the schema documented in the module docstring "
                             "(default: $DATA_DIR/data/corpus/sampler/vs_confirmatory_seeds_final.jsonl, "
                             "the sampler's draw output)")
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--hops", type=int, default=200)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--partial-flush-every", type=int, default=25)
    parser.add_argument("--decoding", choices=("vendor", "default", "locked"),
                        default="vendor",
                        help="vendor = the pinned panel model's recommended "
                             "settings (design default; VENDOR_DECODE table); "
                             "default = no sampling parameters (model "
                             "default); locked = the pilot-matching 0.8/0.5 "
                             "pair (exploratory-style reruns only)")
    parser.add_argument("--temperature", type=float,
                        help="explicit override (needs --top-p; overrides "
                             "--decoding)")
    parser.add_argument("--top-p", type=float,
                        help="explicit override (needs --temperature)")
    parser.add_argument("--limit-seeds", type=int)
    parser.add_argument("--limit-chains", type=int)
    parser.add_argument("--limit-hops", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="chains run concurrently (default 1 = serial). "
                             "Use 16-32 on a single consumer GPU and 64 on an "
                             "A100 node; the binding limit is the serving "
                             "runtime's KV cache, not this client.")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="rerun the chains recorded in FAILED_MSG_* as a "
                             "new replicate (highest failed replicate + 1); "
                             "the original logs are never overwritten")
    parser.add_argument("--verbose", action="store_true",
                        help="print a line per hop flush and per completed "
                             "chain (default: one progress line every 5%% of "
                             "the cell)")
    parser.add_argument("--status", action="store_true",
                        help="report chain status for the logs root and exit "
                             "(read-only: complete / failed / incomplete / "
                             "not started, per task x condition cell)")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="expected seeds per band for --status "
                             "(default: count the seed file)")
    parser.add_argument("--logs-root",
                        help="override the logs root "
                             "(default DATA_DIR/data/vs_data/logs)")
    parser.add_argument("--cache-root",
                        help="override the cache root "
                             "(default DATA_DIR/data/echodrift_confirmatory_cache)")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be 1 or greater")
    if args.rerun_failed and args.replicate:
        parser.error("--rerun-failed chooses the replicate itself "
                     "(highest failed replicate + 1); drop --replicate")
    if args.status:
        return status_report(args)
    if not args.model:
        parser.error("--model is required unless --status is given")

    if args.all_cells:
        cells = [(t, c) for t in TASK_ORDER for c in CONDITION_ORDER]
    elif args.task and args.condition:
        cells = [(args.task, args.condition)]
    else:
        parser.error("give both --task and --condition, or --all-cells")

    seeds_file = Path(args.seeds_file) if args.seeds_file else default_seeds_file()
    band_hint = TASK_BANDS[args.task] if args.task and not args.all_cells else None
    band, seeds = load_seed_file(seeds_file, band=band_hint)
    for task, _ in cells:
        if band != TASK_BANDS[task]:
            parser.error(f"seed band {band!r} does not match task {task!r}'s "
                         f"band {TASK_BANDS[task]!r}")

    load_dotenv()
    if args.logs_root and args.cache_root:
        logs_root = Path(os.path.expanduser(args.logs_root))
        cache_root = Path(os.path.expanduser(args.cache_root))
    else:
        logs_root = (Path(os.path.expanduser(args.logs_root))
                     if args.logs_root
                     else get_data_dir() / "data" / "vs_data" / "logs")
        cache_root = (Path(os.path.expanduser(args.cache_root))
                      if args.cache_root
                      else get_data_dir() / "data" / "echodrift_confirmatory_cache")

    try:
        decoding = resolve_decoding(args.model, args.decoding,
                                    args.temperature, args.top_p)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"decoding for {args.model}: {decoding or 'model default (none sent)'}"
          + (f" [{VENDOR_DECODE_SOURCE.get(args.model, 'explicit')}]"
             if args.temperature is None and args.model in VENDOR_DECODE_SOURCE
             else ""))

    backend = build_backend(args.backend)
    for task, condition in cells:
        run_cell(backend, model=args.model, task=task, condition=condition,
                 band=band, seeds=seeds, logs_root=logs_root,
                 cache_root=cache_root, chains=args.chains, hops=args.hops,
                 replicate=args.replicate,
                 partial_flush_every=args.partial_flush_every,
                 max_retries=args.max_retries,
                 limit_seeds=args.limit_seeds,
                 limit_chains=args.limit_chains,
                 limit_hops=args.limit_hops,
                 decoding=decoding, workers=args.workers,
                 rerun_failed=args.rerun_failed, verbose=args.verbose)
    return 0



if __name__ == "__main__":
    sys.exit(main())