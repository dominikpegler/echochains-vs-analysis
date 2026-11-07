"""Direct-condition transmission prompts for the confirmatory chain generator.

VENDORED COPY (2026-10-31): byte-identical values of the canonical registry
`echochains-core/src/echochain/tasks.py` (TASKS). Only `echochains-core` is
installed in the conda env, so this repo carries a local copy for the
confirmatory generator and for the Vienna-cluster runs, where the sibling
repo is not guaranteed to exist. The two files are deliberately hardcoded,
not imported, and must be kept byte-identical by hand: change one, change
the other in the same commit. `py/test_confirmatory.py` checks the vendored
values against the canonical file whenever the sibling repo is present.

The instruction is joined to the input text with no separator (`build_messages`),
matching the generation and sim repos. Each task is the ordered list of
instruction stages a single hop applies: one stage for the single-instruction
tasks, two for the composite expand+compress hop (the model first enriches,
then condenses, and the selected condensed text is transmitted).

The confirmatory design runs four of these tasks: paraphrase, improve,
take_inspiration, expand+compress ("continue" is NOT in the design; the
registry keeps the key so the sync check stays trivial).
"""

TASKS = {
    "paraphrase": {
        "system": ("You are a large language model. When given a passage, produce "
                   "a fluent, single-paragraph rendition of its content in your own "
                   "words, without quoting it verbatim."),
        "user": ("Take the following text and restate it, preserving meaning but "
                 "using your own wording: "),
    },
    "improve": {
        # Same content as paraphrase, but the model may raise the writing quality,
        # so its stylistic preferences surface more clearly than under the strict
        # meaning-preserving instruction. Sentence seed sets, like paraphrase.
        "system": None,
        "user": ("You will receive a text. Your task is to improve this text: keep "
                 "its meaning and content, but make the writing clearer, more fluent, "
                 "and better structured. Just output the improved text, nothing else. "
                 "Here is the text:"),
    },
    "continue": {
        "system": None,            # single user turn, as in Perez et al.'s released prompts
        "user": ("You will receive a text. Your task is to continue this text. "
                 "Just output your new text, nothing else. Here is the text:"),
    },
    "take_inspiration": {
        "system": None,            # single user turn, as in Perez et al.'s released prompts
        "user": ("You will receive a text. Your task is to create a new original "
                 "text by taking inspiration from this text. Just output your new "
                 "text, nothing else. Here is the text:"),
    },
    "expand+compress": {
        # One hop applies both stages in sequence; the condensed text is carried
        # to the next hop. The stages share the Perez-style single-user format.
        "stages": [
            {"system": None,
             "user": ("You will receive a text. Your task is to enrich this text, "
                      "expanding it with additional detail while preserving its key "
                      "content. Just output the enriched text, nothing else. Here is "
                      "the text:")},
            {"system": None,
             "user": ("You will receive a text. Your task is to shorten this text "
                      "while preserving its key content. Just output the shortened "
                      "text, nothing else. Here is the text:")},
        ],
    },
}

DEFAULT_TASK = "paraphrase"

CONFIRMATORY_TASKS = ("paraphrase", "improve", "take_inspiration", "expand+compress")


def task_stages(task):
    """Return the ordered list of prompt stages for a task.

    Every task has at least one stage; expand+compress has two. Each stage is a
    dict with "system" (possibly None) and "user".
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task!r}; known: {sorted(TASKS)}")
    entry = TASKS[task]
    if "stages" in entry:
        return entry["stages"]
    return [{"system": entry.get("system"), "user": entry["user"]}]


def build_messages(stage, text):
    """Build chat messages for one instruction stage and one input text.

    The instruction is joined to the text with no separator, matching the
    generation and sim repos. This is the one place the join is defined per
    repo; the copies must stay identical.
    """
    messages = []
    if stage.get("system"):
        messages.append({"role": "system", "content": stage["system"]})
    messages.append({"role": "user", "content": f"{stage['user']}{text}"})
    return messages


def hop_messages(task, text):
    """Messages for one full hop: for a composite task, the stages are chained,
    so this returns the messages for the FIRST stage only. Callers that run a
    composite hop must feed each stage's output into the next via
    `build_messages`; this helper is for the single-instruction tasks.
    """
    stages = task_stages(task)
    if len(stages) != 1:
        raise ValueError(f"{task!r} has {len(stages)} stages; iterate them "
                         f"explicitly with build_messages")
    return build_messages(stages[0], text)