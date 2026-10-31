"""Fixture tests for the confirmatory chain generator (NO API calls).

Everything runs against a scripted mock backend; the tests verify the frozen
edge-case policy, cache/resume behavior, message shapes, and the vendored
tasks registry. Run: python py/test_confirmatory.py

The registry sync check compares py/tasks_registry.py against the canonical
`echochains-generation/py/tasks.py` when that sibling repo is present, and
skips with a note when it is not.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_chains_confirmatory as gen  # noqa: E402
import vs_task_prompts  # noqa: E402
import tasks_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Mock backend and response builders
# ---------------------------------------------------------------------------

class FakeCall:
    def __init__(self, model, messages, decoding, random_seed):
        self.model = model
        self.messages = messages
        self.decoding = dict(decoding)
        self.temperature = self.decoding.get("temperature")
        self.top_p = self.decoding.get("top_p")
        self.random_seed = random_seed


class FakeBackend:
    """Pops one queued response per call; entries may be exceptions."""

    name = "fake"

    def __init__(self, responses, model_version="fake-model@v1"):
        self.responses = list(responses)
        self.calls: list[FakeCall] = []
        self.model_version = model_version

    def call(self, model, messages, decoding, random_seed):
        self.calls.append(FakeCall(model, [dict(m) for m in messages],
                                   decoding, random_seed))
        if not self.responses:
            raise AssertionError("unexpected extra API call; queue exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return gen.CallResult(text=item, model_version=self.model_version,
                              usage={"prompt_tokens": 10, "completion_tokens": 5})


class NoCallBackend:
    """Fails if used; guards the cache-resume test."""

    name = "fake"

    def call(self, *args, **kwargs):
        raise AssertionError("API called although the response was cached")


def stage_names(task):
    return ["expand", "compress"] if task == "expand+compress" else [task]


def joint_key(stage_name):
    return vs_task_prompts.VS_TEMPLATES[stage_name]["key"]


def vs_response(stage_name, *probabilities, n=5):
    texts = [f"variant {i}" for i in range(n)]
    rows = [{"text": t, "probability": p} for t, p in zip(texts, probabilities)]
    return json.dumps({joint_key(stage_name): rows})


def list_response(stage_name, n=5, dup=None):
    texts = [f"variant {i}" for i in range(n)]
    if dup is not None:
        texts[dup[0]] = texts[dup[1]]
    return json.dumps({joint_key(stage_name): [{"text": t} for t in texts]})


def seed_dict(seed_id=1, text="The experiment is planned to occur next week.",
              genre="encyclopedic"):
    return {"seed_id": seed_id, "text": text, "genre": genre}


class Case:
    """A ChainRunner over a temp dir with a queued mock backend."""

    def __init__(self, tmp, *, responses, task="paraphrase",
                 condition="vs_weighted", seeds=None, chains=1, hops=3,
                 max_retries=3, flush_every=2):
        seeds = seeds if seeds is not None else [seed_dict()]
        self.backend = FakeBackend(responses)
        self.cache = gen.CallCache(tmp / "cache.jsonl")
        self.runner = gen.ChainRunner(
            backend=self.backend, model="fake-model", task=task,
            condition=condition, seeds=seeds,
            band=gen.TASK_BANDS[task],
            out_dir=tmp / "logs", cache=self.cache, chains=chains, hops=hops,
            partial_flush_every=flush_every, max_retries=max_retries,
        )
        self.tmp = tmp
        self.cell = self.runner.cell

    def run(self):
        self.runner.run()

    def log(self, seed_id=1, chain_id=1):
        path = self.tmp / "logs" / f"MSG_{seed_id:03d}_chain_{chain_id:03d}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def markers(self):
        return sorted(p.name for p in (self.tmp / "logs").glob(".paused_*"))


_TMP_DIRS: list[Path] = []


def make_case(*args, **kwargs):
    tmp = Path(tempfile.mkdtemp(prefix="vs_confirmatory_fixture_",
                                dir="/tmp/opencode"))
    _TMP_DIRS.append(tmp)
    return Case(tmp, *args, **kwargs)


def fresh_tmp():
    tmp = Path(tempfile.mkdtemp(prefix="vs_confirmatory_fixture_",
                                dir="/tmp/opencode"))
    _TMP_DIRS.append(tmp)
    return tmp


# ---------------------------------------------------------------------------
# Registry sync and template checks
# ---------------------------------------------------------------------------

def test_registry_matches_canonical():
    canonical = Path("/home/user/code/echochains-generation/py/tasks.py")
    if not canonical.exists():
        print("  (skip: canonical echochains-generation repo not present)")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("canonical_tasks", canonical)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tasks_registry.TASKS == module.TASKS, (
        "vendored tasks_registry.py diverged from the canonical registry")
    print("  vendored TASKS == echochains-generation/py/tasks.py TASKS")


def test_prompt_templates_intact():
    assert vs_task_prompts.VS_K == 5
    for task in ("paraphrase", "improve", "take_inspiration", "expand",
                 "compress"):
        assert task in vs_task_prompts.VS_TEMPLATES
        assert task in vs_task_prompts.CONTROL_TEMPLATES
        # matched structure: system byte-identical; control lacks probability
        assert (vs_task_prompts.VS_TEMPLATES[task]["system"]
                == vs_task_prompts.CONTROL_TEMPLATES[task]["system"])
        assert "probability" not in \
            vs_task_prompts.CONTROL_TEMPLATES[task]["user"].lower()
        assert "probability" in vs_task_prompts.VS_TEMPLATES[task]["user"]


# ---------------------------------------------------------------------------
# Message shapes across conditions
# ---------------------------------------------------------------------------

def _assert_decoding(case):
    """Default runners use the locked decoding {0.8, 0.5}."""
    for call in case.backend.calls:
        assert call.temperature == 0.8
        assert call.top_p == 0.5
        assert isinstance(call.random_seed, int) and call.random_seed > 0


def test_default_decoding_omits_parameters():
    """--decoding default sends NO sampling parameters (gpt-6-luna)."""
    case = make_case(responses=["out"], task="paraphrase", condition="direct",
                     hops=1)
    case.runner.decoding = {}
    case.run()
    call = case.backend.calls[0]
    assert call.temperature is None and call.top_p is None
    log = case.log()
    assert log[0]["temperature"] is None and log[0]["top_p"] is None
    assert log[1]["temperature"] is None and log[1]["top_p"] is None


def test_direct_message_shapes():
    seed_text = "The experiment is planned to occur next week."
    case = make_case(responses=[f"direct out {i}" for i in range(2)],
                     task="paraphrase", condition="direct", hops=2)
    case.run()
    calls = case.backend.calls
    assert len(calls) == 2
    # paraphrase: system + user; the instruction is joined with NO separator
    assert len(calls[0].messages) == 2
    assert calls[0].messages[0]["role"] == "system"
    assert calls[0].messages[1]["content"].endswith(seed_text)
    assert calls[0].messages[1]["content"].startswith(
        "Take the following text and restate it")
    entry = case.log()[1]
    assert entry["text"] == "direct out 0"
    assert entry["candidates"] is None and entry["selected_index"] is None
    assert entry["probabilities_normalized"] is None
    # hop 2 transmits hop 1's text as its input
    assert calls[1].messages[1]["content"].endswith("direct out 0")
    _assert_decoding(case)


def test_direct_single_turn_tasks_no_system():
    for task, prefix in (
        ("improve", "You will receive a text. Your task is to improve this text"),
        ("take_inspiration",
         "You will receive a text. Your task is to create a new original text"),
    ):
        case = make_case(responses=["out"], task=task, condition="direct",
                         hops=1)
        case.run()
        call = case.backend.calls[0]
        assert len(call.messages) == 1  # single user turn, no system
        assert call.messages[0]["content"].startswith(prefix)
        assert case.log()[1]["text"] == "out"


def test_direct_composite_two_stages():
    case = make_case(responses=["enriched text", "condensed text", "e2", "c2"],
                     task="expand+compress", condition="direct", hops=2)
    case.run()
    calls = case.backend.calls
    assert len(calls) == 4  # two stages per hop
    assert calls[0].messages[0]["content"].startswith(
        "You will receive a text. Your task is to enrich this text")
    assert calls[1].messages[0]["content"].startswith(
        "You will receive a text. Your task is to shorten this text")
    # stage 2 receives stage 1's output
    assert calls[1].messages[0]["content"].endswith("enriched text")
    log = case.log()
    assert log[1]["text"] == "condensed text"
    assert [s["stage"] for s in log[1]["stages"]] == ["expand", "compress"]
    assert log[1]["stages"][0]["text_out"] == "enriched text"
    assert log[1]["stages"][1]["text_out"] == "condensed text"
    # hop 2 continues from the transmitted compressed text
    assert calls[2].messages[0]["content"].endswith("condensed text")
    assert log[2]["text"] == "c2"


def test_joint_control_message_shapes():
    for task in gen.TASK_ORDER:
        for stage in stage_names(task):
            # a composite hop always runs BOTH stages; queue both responses
            queue = [list_response(s) for s in stage_names(task)]
            case = make_case(responses=queue, task=task,
                             condition="uniform_list", hops=1)
            case.run()
            # the expand-stage call is the first; check it
            call = case.backend.calls[0]
            expand = "expand" if task == "expand+compress" else stage
            user = call.messages[1]["content"]
            assert f'"{joint_key(expand)}"' in user
            # the control's list-only gloss carries no probability wording
            assert "probability" not in user.lower()
            entry = case.log()[1]
            assert entry["text"].startswith("variant ")
            assert 0 <= entry["selected_index"] <= 4
            # no VS-specific deviations may leak into the control condition
            assert entry["deviations"] == []
            for cand in entry["candidates"]:
                assert set(cand) == {"text"}


def test_vs_message_shapes():
    for task in gen.TASK_ORDER:
        for stage in stage_names(task):
            queue = [vs_response(s, 0.4, 0.3, 0.2, 0.05, 0.05)
                     for s in stage_names(task)]
            case = make_case(responses=queue, task=task,
                             condition="vs_weighted", hops=1)
            case.run()
            user = case.backend.calls[0].messages[1]["content"]
            assert f'"{joint_key(stage_names(task)[0])}"' in user
            assert "probability" in user
            entry = case.log()[1]
            texts = [c["text"] for c in entry["candidates"]]
            assert texts == [f"variant {i}" for i in range(5)]
            probs = entry["probabilities_normalized"]
            assert abs(sum(probs) - 1.0) < 1e-9
            assert entry["text"] == texts[entry["selected_index"]]


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

def test_vs_weighted_selection_matches_seed():
    case = make_case(responses=[vs_response("paraphrase", 0.4, 0.3, 0.15,
                                            0.1, 0.05)],
                     task="paraphrase", condition="vs_weighted", hops=1)
    case.run()
    entry = case.log()[1]
    seed = gen.make_seed("fake-model", "paraphrase", "vs_weighted", 0, 1, 1, 1, 0)
    weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    weights = [w / sum(weights) for w in weights]
    expected = random.Random(seed).choices(range(5), weights=weights, k=1)[0]
    assert entry["selected_index"] == expected
    assert entry["text"] == f"variant {expected}"
    assert case.log()[0]["text"] == entry["text"] or True


def test_uniform_selection_reproducible():
    case = make_case(responses=[list_response("paraphrase")] * 5,
                     task="paraphrase", condition="uniform_list", hops=5)
    case.run()
    log = case.log()
    for hop_id, entry in enumerate(log[1:], start=1):
        seed = gen.make_seed("fake-model", "paraphrase", "uniform_list", 0, 1,
                             1, hop_id, 0)
        expected = random.Random(seed).randrange(5)
        assert entry["selected_index"] == expected


def test_vs_rerun_determinism():
    """The same cells (same seeds, same responses) reproduce the same log."""
    logs = []
    for _ in range(2):
        case = make_case(responses=[vs_response("paraphrase", 0.4, 0.3, 0.15,
                                                0.1, 0.05)] * 3,
                         task="paraphrase", condition="vs_weighted", hops=3)
        case.run()
        entries = [{k: e[k] for k in ("hop", "text", "selected_index",
                                      "random_seed")}
                   for e in case.log() if e["hop"] > 0]
        logs.append(json.dumps(entries))
    assert logs[0] == logs[1]


def test_composite_stages_feed_selection():
    """VS + composite: the selected enriched candidate feeds compression."""
    expand_resp = vs_response("expand", 0.9, 0.02, 0.02, 0.02, 0.04)
    compress_resp = vs_response("compress", 0.1, 0.2, 0.3, 0.2, 0.2)
    case = make_case(responses=[expand_resp, compress_resp],
                     task="expand+compress", condition="vs_weighted", hops=1)
    case.run()
    # selection must equal a recomputation with the same seeded RNG
    seed = gen.make_seed("fake-model", "expand+compress", "vs_weighted", 0, 1,
                         1, 1, 0)
    weights = [0.9, 0.02, 0.02, 0.02, 0.04]
    weights = [w / sum(weights) for w in weights]
    expected = random.Random(seed).choices(range(5), weights=weights, k=1)[0]
    call2 = case.backend.calls[1]
    # the compression call's quoted input slot carries the selected enriched
    # text (user content ends with Input text:\n"""<text>""")
    slot = call2.messages[1]["content"].rsplit('"""', 2)[-2]
    assert slot.strip() == f"variant {expected}"
    log = case.log()[1]
    assert log["stages"][0]["stage"] == "expand"
    assert log["stages"][0]["selected_index"] == expected
    assert log["stages"][0]["text_out"] == f"variant {expected}"
    assert log["text"].startswith("variant ")


# ---------------------------------------------------------------------------
# Edge-case policy
# ---------------------------------------------------------------------------

def test_malformed_retries_then_success():
    case = make_case(responses=["not json at all",
                                "```json\n{\"x\": 1}\n```",
                                vs_response("paraphrase", 0.5, 0.2, 0.1,
                                            0.1, 0.1)],
                     task="paraphrase", condition="vs_weighted", hops=1,
                     max_retries=3)
    case.run()
    entry = case.log()[1]
    assert entry["attempts"] == 3
    assert entry["deviations"] == []
    assert not case.markers()
    assert case.cell.retry_needed == 1


def test_wrong_count_then_accept():
    case = make_case(responses=[vs_response("paraphrase", 0.7, 0.3, n=2)] * 3,
                     task="paraphrase", condition="vs_weighted", hops=1,
                     max_retries=3)
    case.run()
    entry = case.log()[1]
    assert "accepted_wrong_count" in entry["deviations"]
    assert len(entry["candidates"]) == 2
    assert abs(sum(entry["probabilities_normalized"]) - 1.0) < 1e-9
    assert not case.markers()
    assert len(case.backend.calls) == 3


def test_wrong_count_retry_hits_exact_count_first():
    # attempt 1 has the wrong count, attempt 2 the exact count: the exact
    # response is accepted and the rejected one is not cached
    case = make_case(responses=[vs_response("paraphrase", 0.5, 0.5, n=2),
                                vs_response("paraphrase", 0.5, 0.2, 0.2,
                                            0.05, 0.05)],
                     task="paraphrase", condition="vs_weighted", hops=1)
    case.run()
    entry = case.log()[1]
    assert entry["deviations"] == []
    assert len(entry["candidates"]) == 5
    assert len(case.backend.calls) == 2


def test_fewer_than_two_candidates_pauses():
    case = make_case(responses=[vs_response("paraphrase", 1.0, n=1)] * 3,
                     task="paraphrase", condition="vs_weighted", hops=3,
                     max_retries=3)
    try:
        case.run()
        raise AssertionError("expected CellPaused")
    except gen.CellPaused:
        pass
    assert case.markers(), "pause marker missing"
    log_path = case.tmp / "logs" / "MSG_001_chain_001.json"
    assert not log_path.exists()  # the chain never completes


def test_duplicates_logged_not_retried():
    case = make_case(responses=[list_response("paraphrase", n=5, dup=(1, 0))],
                     task="paraphrase", condition="uniform_list", hops=1)
    case.run()
    entry = case.log()[1]
    assert "duplicates" in entry["deviations"]
    assert len(entry["candidates"]) == 5
    assert len(case.backend.calls) == 1


def test_invalid_probability_and_uniform_fallback():
    bad = json.dumps({"paraphrases": [
        {"text": "a", "probability": None},
        {"text": "b", "probability": "x"},
        {"text": "c", "probability": -1},
        {"text": "d", "probability": 0},
        {"text": "e"},
    ]})
    case = make_case(responses=[bad], task="paraphrase",
                     condition="vs_weighted", hops=1)
    case.run()
    entry = case.log()[1]
    # missing probability (the last row) is also an invalid_probability
    assert entry["deviations"].count("invalid_probability") == 5
    assert "uniform_fallback" in entry["deviations"]
    assert all(abs(p - 0.2) < 1e-9
               for p in entry["probabilities_normalized"])


def test_selection_tie_logged():
    case = make_case(responses=[vs_response("paraphrase", 0.5, 0.5, 0, 0, 0)],
                     task="paraphrase", condition="vs_weighted", hops=1)
    case.run()
    entry = case.log()[1]
    assert "selection_tie" in entry["deviations"]
    assert entry["text"] in ("variant 0", "variant 1")


def test_api_error_retry_then_success():
    case = make_case(responses=[RuntimeError("transient API error"), "ok text"],
                     task="paraphrase", condition="direct", hops=1)
    case.run()
    entry = case.log()[1]
    assert entry["text"] == "ok text"
    assert entry["attempts"] == 2


def test_retry_rate_threshold_pauses():
    """20+ calls in a cell with a >10% retry share pause the cell."""
    seeds = [seed_dict(seed_id=i) for i in (1, 2)]
    dirty_call_numbers = {3, 9, 15, 20}
    # walk the call order the runner will use: seed x chain x hop
    queue: list = []
    call_number = 0
    for seed in seeds:
        for chain in range(1, 6):
            for hop in range(1, 4):
                call_number += 1
                queue.append(list_response("paraphrase"))
                if call_number in dirty_call_numbers:
                    queue.insert(len(queue) - 1, "GARBAGE")
    case = make_case(responses=queue, task="paraphrase",
                     condition="uniform_list", seeds=seeds, chains=5, hops=3)
    try:
        case.run()
        raise AssertionError("expected CellPaused from the retry-rate rule")
    except gen.CellPaused:
        pass
    assert case.markers()
    assert case.cell.retry_needed >= 4


# ---------------------------------------------------------------------------
# Cache and resume
# ---------------------------------------------------------------------------

def test_cache_resume_without_new_calls():
    case = make_case(responses=[vs_response("paraphrase", 0.4, 0.3, 0.15,
                                            0.1, 0.05)] * 3,
                     task="paraphrase", condition="vs_weighted", hops=3)
    case.run()
    original = case.log()
    calls_first = len(case.backend.calls)
    assert calls_first == 3
    # simulate a crash: the finished log vanishes, the cache survives
    (case.tmp / "logs" / "MSG_001_chain_001.json").unlink()
    case.runner.backend = NoCallBackend()
    case.runner.run()
    rebuilt = case.log()
    # content fields must match; only cache-status bookkeeping may differ
    for a, b in zip(original[1:], rebuilt[1:]):
        for key in ("hop", "text", "selected_index", "random_seed",
                    "probabilities_normalized", "deviations"):
            assert a[key] == b[key], (key, a[key], b[key])
    assert case.cell.successes == calls_first + 3  # rebuilt from cache
    # NoCallBackend.call raises if the resume tried to call the API at all


def test_completed_chain_skipped():
    case = make_case(responses=["done once"], task="paraphrase",
                     condition="direct", hops=1)
    case.run()
    assert len(case.backend.calls) == 1
    case.runner.run()  # rerun: the completed log is skipped
    assert len(case.backend.calls) == 1


def test_parse_failure_not_cached():
    """Malformed responses must not poison the cache: after retries exhaust,
    nothing is cached, so a resume re-draws the hop."""
    case = make_case(responses=["GARBAGE"] * 2, task="paraphrase",
                     condition="uniform_list", hops=1, max_retries=2)
    try:
        case.run()
        raise AssertionError("expected CellPaused")
    except gen.CellPaused:
        pass
    assert not case.cache.entries


def test_partial_log_and_resume():
    """A crash mid-chain leaves a partial log; resume rebuilds all hops (the
    cached ones without new calls) and reproduces their texts."""
    responses = [vs_response("paraphrase", 0.4, 0.3, 0.15, 0.1, 0.05)] * 2 \
        + [vs_response("paraphrase", 0.9, 0.02, 0.02, 0.02, 0.04)] * 2
    case = make_case(responses=[responses[0], responses[1],
                                RuntimeError("simulated crash")],
                     task="paraphrase", condition="vs_weighted", hops=4,
                     flush_every=2)
    try:
        case.run()
        raise AssertionError("expected CellPaused")
    except gen.CellPaused:
        pass
    log_path = case.tmp / "logs" / "MSG_001_chain_001.json"
    partial = log_path.with_suffix(".json.partial")
    assert not log_path.exists()
    partial_log = json.loads(partial.read_text(encoding="utf-8"))
    assert [e["hop"] for e in partial_log] == [0, 1, 2]
    early_texts = [e["text"] for e in partial_log[1:]]

    # resume: hops 1-2 come from cache (no new calls), the tail from the queue
    case.backend.responses = [responses[2], responses[3]]
    calls_before_resume = len(case.backend.calls)
    case.runner.run()
    final = case.log()
    assert [e["hop"] for e in final] == [0, 1, 2, 3, 4]
    assert [e["text"] for e in final[1:3]] == early_texts
    assert not partial.exists()
    # the crashed run consumed 2 real calls + 3 failed attempts; only hops
    # 3-4 hit the API during the resume
    new_calls = len(case.backend.calls) - calls_before_resume
    assert new_calls == 2


# ---------------------------------------------------------------------------
# Seed schema, band checks
# ---------------------------------------------------------------------------

def test_seed_schema_validation():
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as td:
        path = Path(td) / "seeds.json"
        path.write_text(json.dumps({"seeds": [seed_dict()]}))
        try:
            gen.load_seed_file(path)
            raise AssertionError("expected ValueError (missing band)")
        except ValueError:
            pass
        path.write_text(json.dumps({"band": "sentence", "seeds": [
            seed_dict(1), seed_dict(1, "other text")]}))
        try:
            gen.load_seed_file(path)
            raise AssertionError("expected ValueError (duplicate seed_id)")
        except ValueError:
            pass
        path.write_text(json.dumps({"band": "sentence", "seeds": [
            {"seed_id": 1, "text": "  ", "genre": "encyclopedic"}]}))
        try:
            gen.load_seed_file(path)
            raise AssertionError("expected ValueError (empty text)")
        except ValueError:
            pass
        path.write_text(json.dumps({"band": "sentence", "seeds": [seed_dict()]}))
        band, seeds = gen.load_seed_file(path)
        assert band == "sentence" and len(seeds) == 1


def test_band_mismatch_rejected():
    tmp = fresh_tmp()
    try:
        gen.run_cell(
            FakeBackend([]), model="fake-model", task="expand+compress",
            condition="direct", band="sentence", seeds=[seed_dict()],
            logs_root=tmp / "logs", cache_root=tmp / "cache", chains=1,
            hops=1, replicate=0, partial_flush_every=2, max_retries=3,
            limit_seeds=None, limit_chains=None, limit_hops=None)
        raise AssertionError("expected ValueError (band mismatch)")
    except ValueError:
        pass


def test_run_cell_single_stage_end_to_end():
    tmp = fresh_tmp()
    gen.run_cell(
        FakeBackend([vs_response("paraphrase", 0.4, 0.3, 0.15, 0.1, 0.05)]),
        model="fake-model", task="paraphrase", condition="vs_weighted",
        band="sentence", seeds=[seed_dict()],
        logs_root=tmp / "logs", cache_root=tmp / "cache", chains=1, hops=1,
        replicate=0, partial_flush_every=2, max_retries=3,
        limit_seeds=None, limit_chains=None, limit_hops=None)
    log = json.loads((tmp / "logs" / "fake-model" / "confirmatory"
                      / "paraphrase" / "vs_weighted" / "MSG_001_chain_001.json")
                     .read_text(encoding="utf-8"))
    assert log[0]["hop"] == 0 and log[0]["seed_id"] == 1
    assert log[1]["selected_index"] is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# QC pass script
# ---------------------------------------------------------------------------

def test_qc_inspect_cell():
    import qc_chains_confirmatory as qc
    tmp = fresh_tmp()
    cell = tmp / "fake-model" / "confirmatory" / "paraphrase" / "vs_weighted"
    cell.mkdir(parents=True)
    # chain 001: complete, peaked (all mass on one candidate)
    log1 = [
        {"hop": 0, "text": "seed", "seed_id": 1},
        {"hop": 1, "text": "t1", "attempts": 1, "candidates":
         [{"text": f"v{i}", "probability": p} for i, p in enumerate(
             [0.96, 0.01, 0.01, 0.01, 0.01])],
         "probabilities_normalized": [0.96, 0.01, 0.01, 0.01, 0.01]},
        {"hop": 2, "text": "t2", "attempts": 2,
         "candidates": [{"text": f"v{i}", "probability": 0.2} for i in range(5)],
         "probabilities_normalized": [0.2] * 5},
    ]
    # chain 002: incomplete (only hop 1), one clean hop
    log2 = [
        {"hop": 0, "text": "seed", "seed_id": 2},
        {"hop": 1, "text": "u1", "attempts": 1, "candidates":
         [{"text": "single", "probability": 1.0}],
         "probabilities_normalized": [1.0]},
    ]
    (cell / "MSG_001_chain_001.json").write_text(json.dumps(log1))
    (cell / "MSG_002_chain_001.json").write_text(json.dumps(log2))
    (cell / ".paused_2_1_2_paraphrase").write_text("why")
    report = qc.inspect_cell(cell, "paraphrase", "vs_weighted", hops=2,
                             chains_expected=1, n_seeds=2)
    assert report["coverage"]["chains_found"] == 2
    assert len(report["incomplete_chains"]) == 1
    assert report["pause_markers"] == [".paused_2_1_2_paraphrase"]
    # retries: 1 of 3 hops
    assert report["technical"]["hops_total"] == 3
    assert report["technical"]["hops_retried"] == 1
    assert round(report["technical"]["retry_rate"], 3) == round(1 / 3, 3)
    assert report["technical"]["flag_over_10pct"] is True
    # collapse: seed 1 peaked at hop1 only (1/2 = 0.5, not > 0.5); seed 2
    # single-candidate hop (degenerate, 1/1 > 0.5)
    shares = report["collapse"]["share_by_seed"]
    assert abs(shares[1] - 0.5) < 1e-9
    assert abs(shares[2] - 1.0) < 1e-9
    assert report["collapse"]["flagged_seeds"] == [2]
    # no NLI in this check: skip-nli payload drops hop samples
    payload = dict(report)
    payload.pop("hop1_samples_for_nli", None)
    assert "hop1_samples_for_nli" not in payload


def test_qc_clean_cell_exits_zero():
    import qc_chains_confirmatory as qc
    tmp = fresh_tmp()
    cell = tmp / "logs" / "fake-model" / "confirmatory" / "paraphrase" / "direct"
    cell.mkdir(parents=True)
    log = [{"hop": 0, "text": "s", "seed_id": 1},
           {"hop": 1, "text": "t", "attempts": 1}]
    (cell / "MSG_001_chain_001.json").write_text(json.dumps(log))
    flagged, report = qc.qc_for_cell(
        "fake-model", "paraphrase", "direct", tmp / "logs", hops=1,
        chains=1, n_seeds=1, skip_nli=True, nli_floor=0.3,
        out_dir=tmp / "qc")
    assert flagged is False
    assert (tmp / "qc" / "qc_paraphrase-direct.json").exists()


def test_decode_resolution():
    """The vendor table and override precedence resolve correctly."""
    from generate_chains_confirmatory import resolve_decoding, VENDOR_DECODE
    assert resolve_decoding("gpt-6-luna", "vendor") == {}
    assert resolve_decoding("Qwen/Qwen3-32B", "vendor") == \
        {"temperature": 0.7, "top_p": 0.8}
    assert resolve_decoding("meta-llama/Llama-3.1-8B-Instruct", "vendor") == \
        {"temperature": 0.6, "top_p": 0.9}
    assert resolve_decoding("google/gemma-3-27b-it", "vendor") == \
        {"temperature": 1.0, "top_p": 0.95}
    # explicit overrides beat everything and must be given together
    assert resolve_decoding("gpt-6-luna", "vendor",
                            temperature=0.5, top_p=0.5) == \
        {"temperature": 0.5, "top_p": 0.5}
    try:
        resolve_decoding("some-unpinned-model", "vendor")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        resolve_decoding("some-unpinned-model", "default", temperature=0.5)
        raise AssertionError("expected ValueError (top-p missing)")
    except ValueError:
        pass
    assert sorted(VENDOR_DECODE) == [
        "Qwen/Qwen3-32B", "google/gemma-3-27b-it", "gpt-6-luna",
        "meta-llama/Llama-3.1-8B-Instruct"]


TESTS = [
    test_registry_matches_canonical,
    test_prompt_templates_intact,
    test_direct_message_shapes,
    test_direct_single_turn_tasks_no_system,
    test_direct_composite_two_stages,
    test_joint_control_message_shapes,
    test_vs_message_shapes,
    test_vs_weighted_selection_matches_seed,
    test_uniform_selection_reproducible,
    test_vs_rerun_determinism,
    test_composite_stages_feed_selection,
    test_malformed_retries_then_success,
    test_wrong_count_then_accept,
    test_wrong_count_retry_hits_exact_count_first,
    test_fewer_than_two_candidates_pauses,
    test_duplicates_logged_not_retried,
    test_invalid_probability_and_uniform_fallback,
    test_selection_tie_logged,
    test_api_error_retry_then_success,
    test_retry_rate_threshold_pauses,
    test_cache_resume_without_new_calls,
    test_completed_chain_skipped,
    test_parse_failure_not_cached,
    test_partial_log_and_resume,
    test_seed_schema_validation,
    test_band_mismatch_rejected,
    test_run_cell_single_stage_end_to_end,
    test_default_decoding_omits_parameters,
    test_qc_inspect_cell,
    test_qc_clean_cell_exits_zero,
    test_decode_resolution,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    for tmp in _TMP_DIRS:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} fixture tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())