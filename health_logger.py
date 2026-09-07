import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/user/data_export/echo-chain/data/echodrift_stats")
LOG = Path("/home/user/health_log.csv")


def shard_counts():
    out = []
    for sub in ["vs_argmax", "vs_weighted"]:
        base = ROOT / sub / "shards" / "gpt-4.1-nano" / sub
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir():
                n = len(list(d.rglob("*.parquet")))
                out.append(f"{d.name}={n}")
    return " ".join(out)


def main():
    while True:
        ts = time.strftime("%F %T")
        try:
            load = os.getloadavg()[0]
            with open("/proc/meminfo") as f:
                mi = {}
                for line in f:
                    if ":" in line:
                        key, vals = line.split(":", 1)
                        mi[key] = int(vals.split()[0])
            mem_used = (mi["MemTotal"] - mi["MemAvailable"]) / 1048576
            swap_used = (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1048576
            with open("/proc/pressure/memory") as f:
                psi_line = [l for l in f if l.startswith("some")][0]
            psi_avg10 = psi_line.split()[1]
            nv = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().replace(", ", "/")
            counts = shard_counts()
            row = (f"{ts} load={load:.1f} mem_used={mem_used:.1f}G "
                   f"swap_used={swap_used:.1f}G psi_mem={psi_avg10} gpu={nv} {counts}")
        except Exception as e:
            row = f"{ts} LOGGER_ERROR {e!r}"
        with open(LOG, "a") as f:
            f.write(row + "\n")
        time.sleep(300)


if __name__ == "__main__":
    main()
