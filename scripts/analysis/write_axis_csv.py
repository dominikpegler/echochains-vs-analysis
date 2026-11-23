import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# The shard workers occupy the GPU; the axis computation only needs the anchor
# centroids, so use a CPU SentenceTransformer. Patch before importing
# semantic_axes, which instantiates the embedding model at import time.
import echochain.embeddings as _em
from sentence_transformers import SentenceTransformer


def _cpu_model():
    return SentenceTransformer("all-mpnet-base-v2", device="cpu")


_em.get_embedding_model = _cpu_model

from echochain.embeddings import load_emb  # noqa: E402
from echochain.semantic_axes import AXES, project_onto_axis  # noqa: E402
from echochain.utils import get_data_dir  # noqa: E402

SUB_PATH = sys.argv[1]  # vs_weighted | vs_argmax
ANALYSIS_NAME = "descriptives_vs" if SUB_PATH == "vs_weighted" else "descriptives_vs_argmax"

DATA_DIR = get_data_dir()
STATS_DIR = Path(DATA_DIR, "data", "echodrift_stats", SUB_PATH)
EMB_DIR = Path(DATA_DIR, "data", "echodrift_emb", ANALYSIS_NAME)

axis_data = []
for model in sorted(os.listdir(EMB_DIR)):
    for config in sorted(os.listdir(EMB_DIR / model)):
        print(model, config, flush=True)
        temp, topp = config.replace("temp", "").replace("topp", "").split("_")
        temp = float(temp)
        topp = float(topp)
        for msg_id in tqdm(sorted(os.listdir(EMB_DIR / model / config))):
            for fname in sorted(os.listdir(EMB_DIR / model / config / msg_id)):
                try:
                    chain_id, hop_id = fname.replace("chain_", "").replace("hop_", "").split("_")
                    chain_id = int(chain_id)
                    hop_id = int(hop_id)
                    emb = load_emb(model, config, msg_id, chain_id, hop_id, EMB_DIR)
                    axis_scores = {}
                    for axis_name, (start_vec, end_vec) in AXES.items():
                        axis_scores[axis_name] = project_onto_axis(emb, start_vec, end_vec)
                    row = {
                        "model": model,
                        "config": config,
                        "msg_id": msg_id,
                        "chain": chain_id,
                        "hop": hop_id,
                        "kind": "cumulative",
                        "temperature": temp,
                        "top_p": topp,
                    }
                    row.update(axis_scores)
                    axis_data.append(row)
                except Exception as e:
                    print(model, config, msg_id, chain_id, hop_id, e, flush=True)

df_axis = pd.DataFrame(axis_data)
STATS_DIR.mkdir(parents=True, exist_ok=True)
out = STATS_DIR / "axis_cumulative_long.csv"
df_axis.to_csv(out, index=False)
print(f"wrote {out} with {len(df_axis)} rows", flush=True)
