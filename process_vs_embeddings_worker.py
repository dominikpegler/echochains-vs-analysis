import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

from echochain.embeddings import get_embedding_model
from echochain.utils import get_data_dir

SUB_PATH = sys.argv[1]          # vs_weighted | vs_argmax
ANALYSIS_NAME = sys.argv[2]     # descriptives_vs | descriptives_vs_argmax
MODEL = sys.argv[3] if len(sys.argv) > 3 else "gpt-4.1-nano"
CONFIG = sys.argv[4]

DATA_DIR = get_data_dir()
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")
EMB_DIR = Path(DATA_DIR, "data", "echodrift_emb", ANALYSIS_NAME)

log_dir = LOGS_DIR / MODEL / SUB_PATH / CONFIG
model_obj = get_embedding_model()

done = 0
for fpath in tqdm(sorted(glob(str(log_dir / "*.json")))):
    msg_id = Path(fpath).stem[:7]
    chain_idx = int(Path(fpath).stem[-3:])
    with open(fpath, "r") as fp:
        chain = json.load(fp)
    for hop in chain:
        outdir = Path(EMB_DIR, MODEL, CONFIG, msg_id)
        os.makedirs(outdir, exist_ok=True)
        outp = Path(outdir, f'chain_{str(chain_idx).zfill(3)}_hop_{hop["hop"]}')
        if outp.exists():
            done += 1
            continue
        emb = model_obj.encode(hop["text"], convert_to_tensor=False)
        np.savetxt(outp, emb)
        done += 1
print(f"embedding worker done: {done} files present for {MODEL}/{CONFIG}", flush=True)
