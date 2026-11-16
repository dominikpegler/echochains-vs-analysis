import subprocess
import sys
import time

PY = "/home/user/miniconda3/envs/echochain/bin/python"
LAUNCHER = "/home/user/code/echochains-vs-analysis/launch_shards.py"


def main():
    while True:
        try:
            subprocess.run([PY, LAUNCHER], cwd="/home/user/code/echochains-vs-analysis")
        except Exception as e:
            print("watchdog error:", repr(e), flush=True)
        time.sleep(180)


if __name__ == "__main__":
    main()
