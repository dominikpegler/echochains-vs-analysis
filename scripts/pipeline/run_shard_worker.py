import sys
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

print(f"=== writing shards {SUB_PATH}/{CONFIG} ===", flush=True)
write_shards(
    model_filter=MODEL,
    config_filter=CONFIG,
    logs_dir=LOGS_DIR,
    sub_path=SUB_PATH,
    emb_dir=EMB_DIR,
    shard_root=SHARD_ROOT,
    metrics_version=METRICS_VERSION,
)
print(f"=== worker done {SUB_PATH}/{CONFIG} ===", flush=True)
