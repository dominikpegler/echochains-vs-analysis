# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # EchoChain with Verbalized Sampling

# %%
import jupyter_black

jupyter_black.load()

# %% [markdown]
# ## 1. Setup

# %%
import os
import json
import time
import re
import random
import hashlib
import struct

from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI as OpenAI
from echochain.utils import (
    find_element_with_hop_value,
    contains_hop_with_value,
    get_data_dir,
)

load_dotenv()

API_KEY = os.environ["API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]

DATA_DIR = get_data_dir()
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")

with open("seed_sentences_50.json", "r") as fp:
    SEED_SENTENCES = json.load(fp)

SYSTEM_PROMPT_DIRECT = (
    "You are a large language model. When given a passage, produce a fluent, "
    "single-paragraph rendition of its content in your own words, without "
    "quoting it verbatim."
)

USER_PROMPT_DIRECT = "Take the following text and restate it, preserving meaning but using your own wording: "

SYSTEM_PROMPT_VS = (
    "You are a large language model. "
    "Given an input text, generate several distinct paraphrases that preserve its meaning "
    "while using different wording. "
    "Return ONLY valid JSON."
)

USER_PROMPT_VS_TEMPLATE = """Take the following text and generate {k} distinct paraphrases that preserve its meaning but use different wording.

For each paraphrase, provide:
1. "text": the paraphrase
2. "probability": a number between 0 and 1 representing how likely this paraphrase is relative to other plausible paraphrases of the same input

The probabilities do not need to sum exactly to 1, but they should reflect relative likelihoods.

Return ONLY JSON in exactly this format:
{{
  "paraphrases": [
    {{"text": "first paraphrase", "probability": 0.42}},
    {{"text": "second paraphrase", "probability": 0.27}}
  ]
}}

Input text:
\"\"\"{text_in}\"\"\"
"""

# %%
MAX_SIGNED_64 = (1 << 63) - 1


def make_seed(
    sentence_id: int,
    chain_id: int,
    hop_id: int,
    replicate: int,
    *,
    salt: bytes | None = None,
) -> int:
    if salt is None:
        salt = os.urandom(8)

    key = f"{sentence_id}:{chain_id}:{hop_id}:{replicate}".encode() + salt
    digest = hashlib.sha256(key).digest()
    seed = struct.unpack(">Q", digest[:8])[0]
    return seed & MAX_SIGNED_64


def normalize_probabilities(candidates):
    probs = [max(0.0, float(c["probability"])) for c in candidates]
    total = sum(probs)
    if total <= 0:
        n = len(candidates)
        return [1.0 / n] * n
    return [p / total for p in probs]


def safe_json_loads(text: str):
    """
    Attempt to extract JSON object from model output.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model output.")


def parse_vs_response(raw_text: str, expected_k: int):
    data = safe_json_loads(raw_text)

    if "paraphrases" not in data or not isinstance(data["paraphrases"], list):
        raise ValueError("JSON missing 'paraphrases' list.")

    candidates = []
    for item in data["paraphrases"]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        prob = item.get("probability", None)

        if not text:
            continue

        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = 0.0

        candidates.append({"text": text, "probability": prob})

    if len(candidates) == 0:
        raise ValueError("No valid paraphrases found in VS response.")

    return candidates[:expected_k]


def select_candidate(candidates, method: str, rng: random.Random):
    if method == "vs_argmax":
        idx = max(range(len(candidates)), key=lambda i: candidates[i]["probability"])
        return idx, candidates[idx]["text"]

    elif method == "vs_uniform":
        idx = rng.randrange(len(candidates))
        return idx, candidates[idx]["text"]

    elif method == "vs_weighted":
        weights = normalize_probabilities(candidates)
        idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        return idx, candidates[idx]["text"]

    else:
        raise ValueError(f"Unknown VS selection method: {method}")


# %% [markdown]
# ## 2. Config

# %%
MODEL = "gpt-4.1-nano"
N_CHAINS = 9
HOPS = 200
GRID = {"temps": [0.4, 0.8, 1.0], "top_ps": [0.3, 0.5, 0.8]}

METHODS = [
    # "direct", # there is already a dedicated script generate_chains.py/ipynb for this method
    "vs_weighted",
    "vs_argmax",
    # "vs_uniform",  # optional
]

VS_K = 5
REPLICATION_ID = 0
SALT = b"iwantitreproducible"

client = OpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# %% [markdown]
# ## 3. Core functions

# %%
def next_hop_direct(
    text_in: str,
    msg_id: int,
    chain_id: int,
    hop_id: int,
    replication_id: int = 0,
    *,
    temperature: float,
    top_p: float,
):
    random_seed = make_seed(msg_id, chain_id, hop_id, replication_id, salt=SALT)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
            {"role": "user", "content": USER_PROMPT_DIRECT + text_in},
        ],
        seed=random_seed,
        temperature=temperature,
        top_p=top_p,
    )

    text_out = resp.choices[0].message.content.strip()
    return {
        "text_out": text_out,
        "random_seed": random_seed,
        "raw_response": text_out,
        "method": "direct",
    }


def next_hop_vs(
    text_in: str,
    msg_id: int,
    chain_id: int,
    hop_id: int,
    replication_id: int = 0,
    *,
    temperature: float,
    top_p: float,
    selection_method: str,
    k: int = 5,
):
    random_seed = make_seed(msg_id, chain_id, hop_id, replication_id, salt=SALT)

    user_prompt = USER_PROMPT_VS_TEMPLATE.format(k=k, text_in=text_in)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VS},
            {"role": "user", "content": user_prompt},
        ],
        seed=random_seed,
        temperature=temperature,
        top_p=top_p,
    )

    raw_text = resp.choices[0].message.content.strip()
    candidates = parse_vs_response(raw_text, expected_k=k)

    rng = random.Random(random_seed)
    selected_idx, text_out = select_candidate(candidates, selection_method, rng)

    return {
        "text_out": text_out,
        "random_seed": random_seed,
        "raw_response": raw_text,
        "method": selection_method,
        "selected_index": selected_idx,
        "candidates": candidates,
    }


def next_hop(
    text_in: str,
    msg_id: int,
    chain_id: int,
    hop_id: int,
    replication_id: int = 0,
    *,
    temperature: float,
    top_p: float,
    method: str,
):
    if method == "direct":
        return next_hop_direct(
            text_in,
            msg_id,
            chain_id,
            hop_id,
            replication_id,
            temperature=temperature,
            top_p=top_p,
        )

    elif method in {"vs_weighted", "vs_argmax", "vs_uniform"}:
        return next_hop_vs(
            text_in,
            msg_id,
            chain_id,
            hop_id,
            replication_id,
            temperature=temperature,
            top_p=top_p,
            selection_method=method,
            k=VS_K,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


# %% [markdown]
# ## 4. Main loop

# %%
for method in METHODS:
    for temp in GRID["temps"]:
        for top_p in GRID["top_ps"]:

            out_dir = Path(LOGS_DIR, MODEL, method, f"temp{temp}_topp{top_p}")
            os.makedirs(out_dir, exist_ok=True)

            for msg_id, seed in enumerate(SEED_SENTENCES["sentences"], start=1):
                print(
                    f"Method: {method}, Temperature: {temp}, Top-p: {top_p} – Message {msg_id:>3}"
                )

                for chain_id in range(1, N_CHAINS + 1):
                    print(
                        f"Method: {method}, Temperature: {temp}, Top-p: {top_p} – "
                        f"Message {msg_id:>3} – Chain {chain_id:>3} ",
                        end="",
                    )

                    hop_text = seed
                    fname = out_dir / f"MSG_{msg_id:03d}_chain_{chain_id:03d}.json"

                    if os.path.exists(fname):
                        log = json.loads(fname.read_text(encoding="utf-8"))
                        print("exists.", end=" ")
                    else:
                        log = [
                            {
                                "hop": 0,
                                "text": hop_text,
                                "method": method,
                            }
                        ]
                        print("created.", end=" ")

                    for hop_id in range(1, HOPS + 1):
                        if contains_hop_with_value(log, hop_id):
                            existing = find_element_with_hop_value(log, hop_id)
                            if isinstance(existing, dict) and "text" in existing:
                                hop_text = existing["text"]
                            else:
                                # fallback if utility returns just text
                                hop_text = existing
                        else:
                            success = False
                            while not success:
                                try:
                                    result = next_hop(
                                        hop_text,
                                        msg_id,
                                        chain_id,
                                        hop_id,
                                        REPLICATION_ID,
                                        temperature=temp,
                                        top_p=top_p,
                                        method=method,
                                    )

                                    hop_text = result["text_out"]

                                    entry = {
                                        "hop": hop_id,
                                        "text": hop_text,
                                        "random_seed": result["random_seed"],
                                        "method": result["method"],
                                        "raw_response": result["raw_response"],
                                    }

                                    if result["method"] != "direct":
                                        entry["selected_index"] = result["selected_index"]
                                        entry["candidates"] = result["candidates"]

                                    log.append(entry)
                                    time.sleep(0.5)
                                    success = True

                                except Exception as e:
                                    print(f"\nRetrying hop {hop_id} due to error: {e}")
                                    time.sleep(1.0)

                    print(f"Last Hop: {hop_id:>3}")
                    fname.write_text(
                        json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            n_sent = len(SEED_SENTENCES["sentences"])
            print(
                f"Finished method={method}: {n_sent} × {N_CHAINS} ({n_sent * N_CHAINS}) chains × {HOPS} hops."
            )

