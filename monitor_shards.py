import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/data_export/echo-chain/data/echodrift_stats"))


def main():
    for sub in ["vs_argmax", "vs_weighted"]:
        base = ROOT / sub / "shards" / "gpt-4.1-nano" / sub
        print(f"--- {sub} ---")
        for d in sorted(base.iterdir()):
            n = len(list(d.rglob("*.parquet"))) if d.is_dir() else 0
            print(f"  {d.name}: {n}/450")


if __name__ == "__main__":
    main()
