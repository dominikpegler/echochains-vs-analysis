import os
import resource
import subprocess
import sys
import time
from pathlib import Path

SUB_PATH = sys.argv[1]  # vs_weighted | vs_argmax
CONFIG = sys.argv[2]

PY = "/home/user/miniconda3/envs/echochain/bin/python"
WORKER = "/home/user/code/echochains-vs-analysis/run_shard_worker.py"
DATA_DIR = "/home/user/data_export/echo-chain"
OUT_DIR = Path(DATA_DIR, "data", "echodrift_stats", SUB_PATH, "shards", "gpt-4.1-nano", SUB_PATH, CONFIG)
STATUS_FILE = OUT_DIR / ".worker_status"


def n_shards():
    # poison-marked files are permanently un-writable, so count them toward the
    # 450 target: otherwise a config with poison never completes and the driver
    # / watchdog churn forever on finished configs
    n = len(list(OUT_DIR.rglob("*.parquet"))) if OUT_DIR.exists() else 0
    n += len(list(OUT_DIR.rglob("*.parquet.poison")))
    return n


def mark_poison():
    """If the worker died to a signal (SIGSEGV bypasses try/except) and a status
    file names the in-flight file, count the crash; after 2 crashes on the same
    file, write a .poison marker so write_shards skips it permanently."""
    try:
        if STATUS_FILE.exists():
            name = STATUS_FILE.read_text().strip()
            if name and name != "OK":
                crash_file = OUT_DIR / f".crashes_{name.replace(' ', '_')}"
                n = 1
                if crash_file.exists():
                    n = int(crash_file.read_text().strip() or "0") + 1
                crash_file.write_text(str(n))
                print(f"crash detected on {name} (#{n})", flush=True)
                if n >= 2:
                    msg_id, chain_idx = name.split()
                    poison = OUT_DIR / msg_id / f"chain_{int(chain_idx):03d}.parquet.poison"
                    poison.parent.mkdir(parents=True, exist_ok=True)
                    poison.touch()
                    crash_file.unlink(missing_ok=True)
                    print(f"=== poison marked: {poison.name} ===", flush=True)
            STATUS_FILE.write_text("OK")
    except Exception as e:
        print(f"mark_poison error: {e!r}", flush=True)


def preexec():
    # never let a segfault reach systemd-coredump: coredumps peaked at 5-6 GB RAM
    # each and were implicated in both system incidents (Sep-01 MM panic, Sep-02 wedge)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for attempt in range(1, 60):
        n = n_shards()
        if n >= 450:
            print(f"=== done {SUB_PATH}/{CONFIG}: {n}/450 after {attempt-1} attempts ===", flush=True)
            return
        print(f"=== {SUB_PATH}/{CONFIG} attempt {attempt}: {n}/450 ===", flush=True)
        r = subprocess.run([PY, WORKER, SUB_PATH, CONFIG], cwd="/home/user/code/echochains-vs-analysis", preexec_fn=preexec)
        mark_poison()
        n_now = n_shards()
        if n_now >= 450:
            print(f"=== done {SUB_PATH}/{CONFIG}: {n_now}/450 after attempt {attempt} ===", flush=True)
            return
        print(f"worker rc={r.returncode}, {n_now}/450, retrying", flush=True)
        time.sleep(2)
    print(f"=== FAILED {SUB_PATH}/{CONFIG} after 60 attempts ===", flush=True)


if __name__ == "__main__":
    main()
