import sys
import time
from pathlib import Path

from echochain.generate_stats_vs import write_shards
from echochain.utils import get_data_dir

SUB_PATH = sys.argv[1]      # vs_weighted | vs_argmax
CONFIG = sys.argv[2]
ANALYSIS_NAME = "descriptives_vs" if SUB_PATH == "vs_weighted" else "descriptives_vs_argmax"
METRICS_VERSION = "v1.0.0"
MODEL = "gpt-4.1-nano"

DATA_DIR = get_data_dir()
LOGS_DIR = Path(DATA_DIR, "data", "echodrift_logs")
EMB_DIR = Path(DATA_DIR, "data", "echodrift_emb", ANALYSIS_NAME)
STATS_DIR = Path(DATA_DIR, "data", "echodrift_stats", SUB_PATH)
SHARD_ROOT = STATS_DIR / "shards"
OUT_DIR = SHARD_ROOT / MODEL / SUB_PATH / CONFIG


def n_shards():
    return len(list(OUT_DIR.glob("*.parquet"))) if OUT_DIR.exists() else 0


def main():
    for attempt in range(1, 50):
        n = n_shards()
        if n >= 450:
            print(f"=== done {SUB_PATH}/{CONFIG}: {n}/450 shards ===", flush=True)
            return
        print(f"=== {SUB_PATH}/{CONFIG} attempt {attempt}: {n}/450 ===", flush=True)
        try:
            write_shards(
                model_filter=MODEL,
                config_filter=CONFIG,
                logs_dir=LOGS_DIR,
                sub_path=SUB_PATH,
                emb_dir=EMB_DIR,
                shard_root=SHARD_ROOT,
                metrics_version=METRICS_VERSION,
            )
        except Exception as e:
            print(f"  ! exception: {e!r}", flush=True)
        time.sleep(1)
    print(f"=== FAILED {SUB_PATH}/{CONFIG} after 50 attempts ===", flush=True)


if __name__ == "__main__":
    main()
