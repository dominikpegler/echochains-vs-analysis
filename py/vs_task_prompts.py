"""VS candidate-elicitation templates for the confirmatory tasks (LOCKED).

Locked wording 2026-10-30, user review. The VS condition of the confirmatory
study crosses all three tasks (paraphrase, take inspiration, expand+compress),
so every task stage needs a candidate-elicitation template: one call per hop
stage, k candidates plus verbalized probabilities, JSON only. The VS-weighted
and verbalized-mode conditions share the call text and differ ONLY in the
selection rule, exactly as in the exploratory pilot
(generate_chains_vs.py: SYSTEM_PROMPT_VS / USER_PROMPT_VS_TEMPLATE).

Direct wordings live in the canonical registry
echochains-generation/py/tasks.py (TASKS); every template here restates the
same task instruction in the VS candidate form that the pilot validated on
paraphrase. JSON keys differ per task ("paraphrases", "new_texts",
"enriched_texts", "shortened_texts"); parsers must select on the template's
key. The paraphrase entry reproduces the pilot texts byte-for-byte.

Run `python py/vs_task_prompts.py` to print all four full texts (prereg
quoting) and run the self-checks.
"""

VS_K = 5

# The boilerplate every template shares after its task sentence: the
# probability gloss, the JSON shape, and the quoted input-text slot.
# Placeholders use @MARKER@ tokens, NEVER str.format fields: the templates
# must keep literal {text_in} and {k} slots plus doubled JSON braces so the
# caller's single .format(k=..., text_in=...) pass works as in the pilot.
_PROB_GLOSS = (
    'For each @NOUN@, provide:\n'
    '1. "text": the @NOUN@\n'
    '2. "probability": a number between 0 and 1 representing how likely this '
    '@NOUN@ is relative to other plausible @NOUNS@ of the same input\n'
    '\n'
    'The probabilities do not need to sum exactly to 1, but they should '
    'reflect relative likelihoods.\n'
    '\n'
    'Return ONLY JSON in exactly this format:\n'
    '{{\n'
    '  "@KEY@": [\n'
    '    {{"text": "first @NOUN@", "probability": 0.42}},\n'
    '    {{"text": "second @NOUN@", "probability": 0.27}}\n'
    '  ]\n'
    '}}\n'
    '\n'
    'Input text:\n"""{text_in}"""'
)


def _build_user(task_sentence, noun, nouns, key):
    gloss = (
        _PROB_GLOSS.replace("@NOUN@", noun).replace("@NOUNS@", nouns)
        .replace("@KEY@", key)
    )
    return task_sentence + "\n\n" + gloss


# task_sentence MUST carry a literal {k} slot and every user template a
# literal {text_in} slot; _build_user preserves them for .format() later.
_TEMPLATES = {
    "paraphrase": {
        "system": (
            "You are a large language model. Given an input text, generate "
            "several distinct paraphrases that preserve its meaning while "
            "using different wording. Return ONLY valid JSON."
        ),
        "task_sentence": (
            "Take the following text and generate {k} distinct paraphrases "
            "that preserve its meaning but use different wording."
        ),
        "noun": "paraphrase",
        "nouns": "paraphrases",
        "key": "paraphrases",
        "stages": 1,
        "source": "exploratory pilot (generate_chains_vs.py), unchanged",
    },
    "take_inspiration": {
        "system": (
            "You are a large language model. Given an input text, generate "
            "several distinct new texts inspired by it. Return ONLY valid "
            "JSON."
        ),
        "task_sentence": (
            "Take the following text and generate {k} distinct new texts "
            "inspired by it."
        ),
        "noun": "text",
        "nouns": "texts",
        "key": "new_texts",
        "stages": 1,
        "source": "locked 2026-10-30; restates the Perez single-turn task",
    },
    "expand": {
        "system": (
            "You are a large language model. Given an input text, generate "
            "several distinct enriched versions of it that expand it with "
            "additional detail while preserving its key content. Return ONLY "
            "valid JSON."
        ),
        "task_sentence": (
            "Take the following text and generate {k} distinct enriched "
            "versions of it that expand it with additional detail while "
            "preserving its key content."
        ),
        "noun": "enriched version",
        "nouns": "enriched versions",
        "key": "enriched_texts",
        "stages": 1,
        "source": "locked 2026-10-30; stage 1 of the expand+compress hop",
    },
    "compress": {
        "system": (
            "You are a large language model. Given an input text, generate "
            "several distinct shortened versions of it that preserve its key "
            "content. Return ONLY valid JSON."
        ),
        "task_sentence": (
            "Take the following text and generate {k} distinct shortened "
            "versions of it that preserve its key content."
        ),
        "noun": "shortened version",
        "nouns": "shortened versions",
        "key": "shortened_texts",
        "stages": 1,
        "source": "locked 2026-10-30; stage 2 of the expand+compress hop",
    },
}

# Exact user templates, built once at import so run code and printed
# prereg text are guaranteed identical.
VS_TEMPLATES = {
    name: {
        "system": spec["system"],
        "user": _build_user(
            spec["task_sentence"], spec["noun"], spec["nouns"], spec["key"]
        ),
        "key": spec["key"],
        "stages": spec["stages"],
        "source": spec["source"],
    }
    for name, spec in _TEMPLATES.items()
}


def user_template(task):
    """Exact user template for one task (unformatted; keeps {k}/{text_in})."""
    return VS_TEMPLATES[task]["user"]


def render(task, text_in, k=VS_K):
    """Chat messages for one VS candidate-elicitation call.

    Returns the ordered message dicts with the system prompt first. The
    result still needs the caller's client call; this module knows nothing
    about endpoints or selection rules.
    """
    spec = VS_TEMPLATES[task]
    return [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": spec["user"].format(k=k, text_in=text_in)},
    ]


if __name__ == "__main__":
    sample = "The experiment is planned to occur next week."
    for name, spec in VS_TEMPLATES.items():
        print(f"### {name} ({spec['source']})\n")
        print(f"system:\n{spec['system']}\n")
        print(f"user (k={VS_K}, text_in={sample!r}):\n")
        print(render(name, sample)[1]["content"])
        print("\n" + "-" * 72 + "\n")
    # Self-checks: slot preservation in the raw templates, slot resolution
    # after formatting, per-task JSON keys present exactly twice (header +
    # sample rows), and the paraphrase texts matching the pilot's literals.
    pilot_system = (
        "You are a large language model. Given an input text, generate "
        "several distinct paraphrases that preserve its meaning while using "
        "different wording. Return ONLY valid JSON."
    )
    pilot_probe = "generate {k} distinct paraphrases that preserve its meaning but use different wording."
    assert VS_TEMPLATES["paraphrase"]["system"] == pilot_system
    assert pilot_probe in VS_TEMPLATES["paraphrase"]["user"]
    for name, spec in VS_TEMPLATES.items():
        raw = spec["user"]
        assert "{k}" in raw and "{text_in}" in raw, name
        assert "@" not in raw, name
        msg = render(name, sample)
        assert len(msg) == 2
        assert f'"{spec["key"]}"' in msg[1]["content"], name
        assert f"{VS_K} distinct" in msg[1]["content"], name
        assert sample in msg[1]["content"] and "{{" not in msg[1]["content"], name
    print("self-checks passed for all", len(VS_TEMPLATES), "templates")