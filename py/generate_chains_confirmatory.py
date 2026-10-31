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
(Llama-3.1-8B 0.6/0.9, Qwen3-32B non-thinking 0.7/0.8, Gemma-3-27b-it
1.0/0.95; card unverified entries are marked VERIFY AT EXECUTION there).
--decoding vendor (the design default) resolves per model; --decoding
default sends no sampling arguments (recorded as model_default); --decoding
locked is the pilot-matching 0.8/0.5 pair for exploratory-style reruns.
The exploratory pilot script `generate_chains_vs.py` stays untouched as the
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
- Retries exhausted (or k' < 2): the stage fails the cell. A stage failure
  is FATAL for the chain (the next hop cannot advance), so the cell pauses:
  a marker file .paused <seed>_<chain>_<hop>_<stage> is written in the cell
  directory and the run exits. There is NO fallback to another condition.
- Separately, the Methods' pause-and-inspect rule: once >= 20 calls in a
  cell needed a retry (parse or API) and their share exceeds 10% of
  completed calls, the cell also PAUSES (retry-rate marker) before failure
  rates can compound further.

RESUME / CACHE:
- A response-level cache (append-only JSONL) keyed on the SHA-256 of backend
  + model + message contents + decoding parameters + API seed. Only responses
  that PARSED cleanly are cached, so a malformed response is always re-drawn
  on resume. A crash resumes a chain from cached responses without
  re-calling the API; selection is reproducible because the RNG is seeded
  from make_seed.
- Chain logs (per seed x chain JSON files) are written when the chain
  completes, mirroring the pilot's layout, and a partial log is flushed every
  --partial-flush-every hops so an unclean crash loses at most that many
  hops. Completed chains are skipped by their log file.

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

SEED FILE SCHEMA (--seeds-file; validated on load):
  {"band": "sentence" | "long", "seeds": [{"seed_id": 1, "text": "...",
    "genre": "encyclopedic" | "fiction" | "dialogue"}, ...]}
  The confirmatory seed sets themselves are built under plan step 5; the
  fixtures in py/test_confirmatory.py use synthetic seeds. The band must
  match the task (expand+compress: long; the other three tasks: sentence).

OUTPUT LAYOUT (under DATA_DIR/data by default, overridable):
  echodrift_logs/<model>/confirmatory/<task>/<condition>/MSG_<seed>_chain_<chain>.json
  echodrift_confirmatory_cache/<backend>-<model>/<task>-<condition>.jsonl

Backends: "azure" (Luna; env API_KEY / AZURE_ENDPOINT / API_VERSION as in
the pilot) and "openai_compatible" (panel models on the Vienna cluster; env
OPENAI_BASE_URL / OPENAI_API_KEY).

Verify before any collection: python py/test_confirmatory.py (fixture
tests, no API calls), then a small live run (--limit-seeds 2 --limit-chains 1
--limit-hops 3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import sys
import time
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
#   Gemma-3:    Google's model card (1.0/0.95); repo gated here, verify the
#               card value at execution.
VENDOR_DECODE = {
    "gpt-6-luna": {},
    "meta-llama/Llama-3.1-8B-Instruct": {"temperature": 0.6, "top_p": 0.9},
    "Qwen/Qwen3-32B": {"temperature": 0.7, "top_p": 0.8},
    "google/gemma-3-27b-it": {"temperature": 1.0, "top_p": 0.95},
}
VENDOR_DECODE_SOURCE = {
    "gpt-6-luna": "vendor constraint (no parameters), verified 2026-10-31",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta model card; VERIFY AT EXECUTION",
    "Qwen/Qwen3-32B": "Qwen model card non-thinking mode; verified 2026-10-31",
    "google/gemma-3-27b-it": "Google Gemma 3 card; VERIFY AT EXECUTION",
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
    """Append-only JSONL response cache (response-level resume)."""

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

    def has(self, key: str) -> bool:
        if key in self.entries:
            self.hits += 1
            return True
        return False

    def put(self, key: str, record: dict) -> None:
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


def extract_json_object(raw: str) -> dict:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
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
    data = extract_json_object(raw)
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

class CellPaused(Exception):
    """A fatal stage failure or the retry-rate threshold; the cell must stop."""


class CellState:
    """Per-cell retry accounting for the pause-and-inspect rule.

    `retry_needed` counts completed calls that needed at least one retry
    (parse or API); once >= 20 calls completed and the retry share exceeds
    RETRY_PAUSE_RATE, the cell pauses BEFORE failure rates can compound.
    """

    RETRY_PAUSE_RATE = 0.10
    RETRY_PAUSE_MIN_CALLS = 20

    def __init__(self, max_retries: int):
        self.max_retries = max_retries
        self.successes = 0
        self.retry_needed = 0

    def after_call(self, attempts: int) -> None:
        self.successes += 1
        if attempts > 1:
            self.retry_needed += 1
        total = self.successes
        if (total >= self.RETRY_PAUSE_MIN_CALLS
                and self.retry_needed / total > self.RETRY_PAUSE_RATE):
            raise CellPaused(
                f"retry rate {self.retry_needed}/{total} exceeds "
                f"{self.RETRY_PAUSE_RATE:.0%}; pause and inspect the cell")


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
    CellPaused when a stage fails fatally or the cell pauses.
    """
    decoding = DECODING["locked"] if decoding is None else dict(decoding)
    if not text_in or not text_in.strip():
        raise CellPaused(f"empty input text at seed {seed_id} chain {chain_id} "
                         f"hop {hop_id} stage {stage_name}")
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
                continue
            except ParseError as exc:
                last_error = exc
                continue
            if "wrong_count" in parse_devs and not from_cache \
                    and attempt < cell.max_retries:
                # a response with the exact count is preferred; retry before
                # accepting the mismatch (frozen policy above)
                last_error = ParseError("wrong candidate count")
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
        raise CellPaused(
            f"stage failed after {cell.max_retries} attempts at seed {seed_id} "
            f"chain {chain_id} hop {hop_id} stage {stage_name}: {last_error}")

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


class ChainRunner:
    """One task x condition cell: iterates seeds and chains, writes logs."""

    def __init__(self, *, backend, model: str, task: str, condition: str,
                 seeds: list[dict], band: str, out_dir: Path, cache: CallCache,
                 chains: int = 5, hops: int = 200, replicate: int = 0,
                 partial_flush_every: int = 25,
                 max_retries: int = 3,
                 vs_k: int = vs_task_prompts.VS_K,
                 decoding: dict | None = None):
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

    # -- helpers ------------------------------------------------------------
    def _chain_path(self, seed_id: int, chain_id: int) -> Path:
        name = f"MSG_{seed_id:03d}_chain_{chain_id:03d}"
        if self.replicate:
            name += f"_rep{self.replicate}"  # QC reruns never clobber the original
        return self.out_dir / f"{name}.json"

    @staticmethod
    def _log_complete(log: list[dict], hops: int) -> bool:
        return bool(log) and any(e.get("hop") == hops for e in log)

    def _pause(self, seed_id: int, chain_id: int, hop_id: int,
               stage_name: str, reason: str) -> None:
        marker = (self.out_dir /
                  f".paused_{seed_id}_{chain_id}_{hop_id}_{stage_name}")
        marker.write_text(reason + "\n", encoding="utf-8")
        print(f"CELL PAUSED: {reason}; marker {marker.name}")
        raise CellPaused(reason)

    def _write(self, log: list[dict], path: Path, final: bool) -> None:
        target = path if final else path.with_suffix(".json.partial")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
        tmp.rename(target)

    # -- chain loop -----------------------------------------------------------
    def run_chain(self, seed: dict, chain_id: int) -> None:
        seed_id = seed["seed_id"]
        path = self._chain_path(seed_id, chain_id)
        if path.exists():
            log = json.loads(path.read_text(encoding="utf-8"))
            if self._log_complete(log, self.hops):
                return
        else:
            log = [{
                "hop": 0, "text": seed["text"], "task": self.task,
                "condition": self.condition, "seed_id": seed_id,
                "band": self.band, "genre": seed.get("genre"),
                "model": self.model, "backend": self.backend.name,
                "temperature": self.decoding.get("temperature"),
                "top_p": self.decoding.get("top_p"),
            }]
        composite = self.task == "expand+compress"
        for hop_id in range(1, self.hops + 1):
            existing = next((e for e in log if e.get("hop") == hop_id), None)
            if existing is not None:
                continue
            self._run_hop(log, seed_id, chain_id, hop_id)
            if hop_id % self.partial_flush_every == 0 and hop_id < self.hops:
                self._write(log, path, final=False)
        self._write(log, path, final=True)
        partial = path.with_suffix(".json.partial")
        if partial.exists():
            partial.unlink()

    def _run_hop(self, log: list[dict], seed_id: int, chain_id: int,
                 hop_id: int) -> str:
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
            try:
                record = run_stage_call(
                    self.backend, self.cache, self.cell, model=self.model,
                    task=self.task, condition=self.condition,
                    stage_name=stage_name, stage_index=stage_index,
                    text_in=text_in, seed_id=seed_id, chain_id=chain_id,
                    hop_id=hop_id, replicate=self.replicate, k=self.vs_k,
                    decoding=self.decoding)
            except CellPaused as exc:
                self._pause(seed_id, chain_id, hop_id, stage_name, str(exc))
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
    def run(self, limit_seeds: int | None = None,
            limit_chains: int | None = None,
            limit_hops: int | None = None) -> None:
        if limit_hops is not None:
            self.hops = min(self.hops, limit_hops)
        chains = self.chains if limit_chains is None else min(self.chains,
                                                              limit_chains)
        seeds = self.seeds if limit_seeds is None else self.seeds[:limit_seeds]
        for seed in seeds:
            for chain_id in range(1, chains + 1):
                self.run_chain(seed, chain_id)
        print(f"cell complete: {self.task} / {self.condition} "
              f"({len(seeds)} seeds x {chains} chains x {self.hops} hops; "
              f"{self.cell.successes} successful calls, "
              f"{self.cell.retry_needed} needed a retry)")


# ---------------------------------------------------------------------------
# Seed loading and CLI
# ---------------------------------------------------------------------------

BANDS = ("sentence", "long")


def load_seed_file(path: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    band = data.get("band")
    if band not in BANDS:
        raise ValueError(f"{path}: missing/unknown band; expected one of {BANDS}")
    seeds = data.get("seeds")
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


def run_cell(backend, *, model: str, task: str, condition: str, band: str,
             seeds: list[dict], logs_root: Path, cache_root: Path,
             chains: int, hops: int, replicate: int, partial_flush_every: int,
             max_retries: int, limit_seeds: int | None,
             limit_chains: int | None, limit_hops: int | None,
             decoding: dict | None = None) -> None:
    expected = TASK_BANDS[task]
    if band != expected:
        raise ValueError(f"seed band {band!r} does not match task {task!r}'s "
                         f"band {expected!r}")
    cache = CallCache(cache_file(cache_root, backend.name, model, task,
                                 condition))
    runner = ChainRunner(
        backend=backend, model=model, task=task, condition=condition,
        seeds=seeds, band=band,
        out_dir=cell_out_dir(logs_root, model, task, condition),
        cache=cache, chains=chains, hops=hops, replicate=replicate,
        partial_flush_every=partial_flush_every, max_retries=max_retries,
        decoding=decoding,
    )
    print(f"=== {task} / {condition} ===")
    runner.run(limit_seeds=limit_seeds, limit_chains=limit_chains,
               limit_hops=limit_hops)
    print(f"cache: {cache.hits} cached-response hits "
          f"({cache.path})")


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
    parser.add_argument("--model", required=True,
                        help="model or deployment identifier (pinned in logs)")
    parser.add_argument("--seeds-file", required=True,
                        help="seed JSON in the schema documented in the module docstring")
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
    parser.add_argument("--logs-root",
                        help="override the logs root "
                             "(default DATA_DIR/data/echodrift_logs)")
    parser.add_argument("--cache-root",
                        help="override the cache root "
                             "(default DATA_DIR/data/echodrift_confirmatory_cache)")
    args = parser.parse_args(argv)

    if args.all_cells:
        cells = [(t, c) for t in TASK_ORDER for c in CONDITION_ORDER]
    elif args.task and args.condition:
        cells = [(args.task, args.condition)]
    else:
        parser.error("give both --task and --condition, or --all-cells")

    band, seeds = load_seed_file(Path(args.seeds_file))
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
                     else get_data_dir() / "data" / "echodrift_logs")
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
    try:
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
                     decoding=decoding)
    except CellPaused as exc:
        # pause-and-inspect: the marker file documents the failing position;
        # rerunning the same command resumes from cache and chain logs.
        print(f"STOPPED (paused cell): {exc}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())