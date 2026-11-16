import os
import re
import subprocess
from pathlib import Path

MAX_PARALLEL = int(os.environ.get("SHARD_MAX_PARALLEL", "2"))

SUB_PATHS = {
    "vs_argmax": [
        "temp0.8_topp0.3",
        "temp0.8_topp0.5",
        "temp0.8_topp0.8",
        "temp1.0_topp0.3",
        "temp1.0_topp0.5",
        "temp1.0_topp0.8",
    ],
    "vs_weighted": [
        "temp1.0_topp0.3",
        "temp1.0_topp0.5",
        "temp1.0_topp0.8",
    ],
}

PY = "/home/user/miniconda3/envs/echochain/bin/python"
RUNNER = "/home/user/code/echochains-vs-analysis/run_shard_driver.py"


def shard_count(sub_path, config):
    DATA_DIR = os.path.expanduser(os.environ.get("DATA_DIR", "~/data_export/echo-chain"))
    p = Path(DATA_DIR, "data", "echodrift_stats", sub_path, "shards", "gpt-4.1-nano", sub_path, config)
    if not p.exists():
        return 0
    # poison-marked files count toward the 450 target: they are permanently
    # un-writable, so counting only parquet makes finished configs respawn forever
    return len(list(p.rglob("*.parquet"))) + len(list(p.rglob("*.parquet.poison")))


def list_screens():
    subprocess.run(["screen", "-wipe"], capture_output=True, text=True)
    out = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
    return set(re.findall(r"(\d+)\.([\w.]+)\s", out))


def launch():
    screens = list_screens()
    alive = 0
    for sub_path, configs in SUB_PATHS.items():
        for cfg in configs:
            name = f"{sub_path}_{cfg}"
            if any(name in s[1] for s in screens):
                print(f"{name}: screen alive, skipping", flush=True)
                alive += 1
                continue
            n = shard_count(sub_path, cfg)
            if n >= 450:
                print(f"{name}: already complete ({n}/450), skipping", flush=True)
                continue
            if alive >= MAX_PARALLEL:
                print(f"{name}: at cap {MAX_PARALLEL}, queued", flush=True)
                continue
            print(f"launching screen {name} ({n}/450)", flush=True)
            subprocess.run(
                ["screen", "-dmS", name, PY, RUNNER, sub_path, cfg],
                cwd="/home/user/code/echochains-vs-analysis",
            )
            alive += 1
    print(f"all launched (cap {MAX_PARALLEL})", flush=True)


if __name__ == "__main__":
    launch()
