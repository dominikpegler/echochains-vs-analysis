import os
import subprocess
import sys
from pathlib import Path

SUB_PATH = sys.argv[1]          # vs_weighted | vs_argmax
ANALYSIS_NAME = sys.argv[2]     # descriptives_vs | descriptives_vs_argmax
MODEL = "gpt-4.1-nano"

DATA_DIR = os.path.expanduser(os.environ.get("DATA_DIR", "~/data_export/echo-chain"))
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")
EMB_DIR = Path(DATA_DIR, "data", "echodrift_emb", ANALYSIS_NAME)

PY = "/home/user/miniconda3/envs/echochain/bin/python"
WORKER = "/home/user/code/echochains-vs-analysis/process_vs_embeddings_worker.py"


def configs_to_do(restrict=None):
    sub = LOGS_DIR / MODEL / SUB_PATH
    todo = []
    for config in sorted(os.listdir(sub)):
        if restrict and config != restrict:
            continue
        log_dir = sub / config
        n_logs = len(list(log_dir.glob("*.json")))
        embdir = EMB_DIR / MODEL / config
        n_emb = len(list(embdir.glob("*/chain_*_hop_*"))) if embdir.exists() else 0
        expected = n_logs * 201  # 201 hops (hop 0..200); 450 chains -> 90450
        todo.append((config, n_emb, expected))
    return todo


def main():
    restrict = sys.argv[3] if len(sys.argv) > 3 else None
    for config, n_emb, expected in configs_to_do(restrict):
        if n_emb >= expected:
            print(f"{config}: complete ({n_emb}/{expected}), skipping", flush=True)
            continue
        print(f"{config}: {n_emb}/{expected}, running", flush=True)
        for attempt in range(1, 30):
            print(f"  attempt {attempt}", flush=True)
            subprocess.run(
                [PY, WORKER, SUB_PATH, ANALYSIS_NAME, MODEL, config],
                cwd="/home/user/code/echochains-vs-analysis",
            )
            embdir = EMB_DIR / MODEL / config
            n_now = len(list(embdir.glob("*/chain_*_hop_*"))) if embdir.exists() else 0
            if n_now >= expected:
                print(f"{config}: complete after attempt {attempt}", flush=True)
                break
            print(f"  worker exited, {n_now}/{expected}, retrying", flush=True)
        else:
            print(f"{config}: FAILED after 30 attempts", flush=True)


if __name__ == "__main__":
    main()
