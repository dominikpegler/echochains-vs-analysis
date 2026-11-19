#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one model and run every confirmatory cell against it, in one job.
#
# This is the shell equivalent of generate_chains_confirmatory.ipynb: serve
# the model with serve_model.sh (inline, inside an existing allocation), wait
# until the OpenAI-compatible endpoint answers, then loop over the four tasks
# and the three conditions and drive the frozen generator for each cell.
#
# Use this to run the whole process inside one Slurm job:
#
#   scripts/confirmatory/serve_model_and_generate.sh \
#       --model Qwen/Qwen3-32B --served-name Qwen3-32B \
#       --tp 2 --dtype bfloat16 --max-model-len 32768 --gpu-mem-util 0.92 \
#       --temperature 0.7 --top-p 0.8 --nworkers 32
#
# The client always talks to http://localhost:$PORT/v1, i.e. the server in
# this job; no tunnel is needed. The endpoint is polled through /v1/models and
# its `id` list must contain the served name before any generation starts.
#
# Decoding: give --temperature and --top-p explicitly. Without them the
# generator falls back to its --decoding vendor table, which only knows the
# full pinned HF ids, so a short --served-name is rejected there. The shipped
# job scripts pass the card values (Qwen3-32B 0.7/0.8, Gemma 4 1.0/0.95).
#
# Resume semantics are the generator's, unchanged: re-running the same command
# skips complete chains, replays accepted calls from the response cache, and
# skips chains recorded in FAILED_MSG_*. Default is a plain resume
# (--replicate 0); --rerun-failed reruns the recorded failed pairs as a new
# replicate instead.
#
# Exit codes: 0 every cell finished without a pause; 3 at least one cell
# paused (resumable: re-run the same command); 1 a cell or the server failed
# hard. --stop-on-pause stops the loop at the first pause/error instead of
# continuing with the remaining cells.
# ---------------------------------------------------------------------------
set -euo pipefail

script_abs="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_abs")/../.." && pwd)"
SERVE_SCRIPT="$repo_root/scripts/confirmatory/serve_model.sh"
GENERATOR="$repo_root/scripts/confirmatory/generate_chains_confirmatory.py"

MODEL=""
SERVED_NAME=""
PORT="${PORT:-8892}"
TP="${TP:-1}"
DTYPE=""
QUANT=""
MAX_MODEL_LEN=""
GPU_MEM_UTIL=""
EXTRA_ARGS=""
API_KEY="${API_KEY:-local}"
NWORKERS=32
RERUN_FAILED=0
STOP_ON_PAUSE=0
CHAINS=5
HOPS=200
MAX_RETRIES=3
PARTIAL_FLUSH_EVERY=25
TEMPERATURE=""
TOP_P=""
SEEDS_FILE=""
LOGS_ROOT=""
CACHE_ROOT=""
PYTHON="${PYTHON:-python3}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

usage() {
  cat <<'EOF'
Usage: serve_model_and_generate.sh --model MODEL [options]

Serve flags (passed to serve_model.sh):
  --model MODEL            HF id or local path (required)
  --served-name NAME       name the API exposes (default: MODEL)
  --port PORT              HTTP port (default 8892)
  --tp N                   tensor-parallel size (default 1)
  --dtype DTYPE            float32 | float16 | bfloat16 (default: vLLM auto)
  --quant QUANT            e.g. bitsandbytes (default: none)
  --max-model-len N        context cap (default: model config)
  --gpu-mem-util F         vLLM default 0.9
  --extra-args "..."       further vllm serve args
  --api-key KEY            value clients send (default: local)

Generation flags (passed to generate_chains_confirmatory.py):
  --nworkers N             chains run concurrently (default 32; an A100
                           node supports this, a consumer GPU 16)
  --rerun-failed           rerun FAILED_MSG_* pairs at a new replicate
                           (default: plain resume at --replicate 0)
  --chains N               chains per seed (default 5)
  --hops N                 hops per chain (default 200)
  --max-retries N          attempts per hop stage (default 3)
  --partial-flush-every N  partial-log interval (default 25)
  --temperature F          decoding temperature (give with --top-p)
  --top-p F                decoding top-p (give with --temperature)
  --seeds-file PATH        default: $DATA_DIR/data/corpus/sampler/
                           vs_confirmatory_seeds_final.jsonl
  --logs-root PATH         default: $DATA_DIR/data/echodrift_logs
  --cache-root PATH        default: $DATA_DIR/data/echodrift_confirmatory_cache
  --python PY              interpreter for the generator and probes
                           (default: python3; the serving env's python)
  --stop-on-pause          stop the cell loop on the first pause/error
  -h|--help                show this help

Environment: API_KEY (client key, default local), SERVER_TIMEOUT (seconds to
wait for the endpoint, default 3600), POLL_INTERVAL (seconds, default 10).
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2 ;;
    --served-name)   SERVED_NAME="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --tp)            TP="$2"; shift 2 ;;
    --dtype)         DTYPE="$2"; shift 2 ;;
    --quant)         QUANT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2 ;;
    --extra-args)    EXTRA_ARGS="$2"; shift 2 ;;
    --api-key)       API_KEY="$2"; shift 2 ;;
    --nworkers)      NWORKERS="$2"; shift 2 ;;
    --rerun-failed)  RERUN_FAILED=1; shift ;;
    --stop-on-pause) STOP_ON_PAUSE=1; shift ;;
    --chains)        CHAINS="$2"; shift 2 ;;
    --hops)          HOPS="$2"; shift 2 ;;
    --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
    --partial-flush-every) PARTIAL_FLUSH_EVERY="$2"; shift 2 ;;
    --temperature)   TEMPERATURE="$2"; shift 2 ;;
    --top-p)         TOP_P="$2"; shift 2 ;;
    --seeds-file)    SEEDS_FILE="$2"; shift 2 ;;
    --logs-root)     LOGS_ROOT="$2"; shift 2 ;;
    --cache-root)    CACHE_ROOT="$2"; shift 2 ;;
    --python)        PYTHON="$2"; shift 2 ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage 1; }
[[ -n "$SERVED_NAME" ]] || SERVED_NAME="$MODEL"
[[ -x "$SERVE_SCRIPT" ]] || { echo "error: serve_model.sh not found at $SERVE_SCRIPT" >&2; exit 1; }
[[ -f "$GENERATOR" ]] || { echo "error: generator not found at $GENERATOR" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: '$PYTHON' not on PATH" >&2; exit 1; }
if [[ -n "$TEMPERATURE" || -n "$TOP_P" ]]; then
  [[ -n "$TEMPERATURE" && -n "$TOP_P" ]] || {
    echo "error: --temperature and --top-p must be given together" >&2; exit 1; }
fi
if [[ -z "$SEEDS_FILE" && -z "${DATA_DIR:-}" ]]; then
  echo "error: DATA_DIR is unset and no --seeds-file given" >&2
  exit 1
fi

TASKS=(paraphrase improve take_inspiration expand+compress)
CONDITIONS=(direct uniform_list vs_weighted)

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# -- phase 1: serve ---------------------------------------------------------
serve_args=(--model "$MODEL" --served-name "$SERVED_NAME" --port "$PORT"
            --tp "$TP" --api-key "$API_KEY")
[[ -n "$DTYPE" ]]         && serve_args+=(--dtype "$DTYPE")
[[ -n "$QUANT" ]]         && serve_args+=(--quant "$QUANT")
[[ -n "$MAX_MODEL_LEN" ]] && serve_args+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$GPU_MEM_UTIL" ]]  && serve_args+=(--gpu-mem-util "$GPU_MEM_UTIL")
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
[[ -n "$EXTRA_ARGS" ]]    && serve_args+=($EXTRA_ARGS)

endpoint="http://localhost:${PORT}/v1"
echo "serving:      $MODEL (as $SERVED_NAME, tp=$TP) on $endpoint"
"$SERVE_SCRIPT" "${serve_args[@]}" &
SERVER_PID=$!

wait_for_server() {
  echo "waiting for the endpoint (timeout ${SERVER_TIMEOUT}s)"
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "error: the server process exited before becoming ready" >&2
      return 1
    fi
    if "$PYTHON" - "$PORT" "$SERVED_NAME" "$API_KEY" <<'PY'
import json, sys, urllib.request
port, name, key = sys.argv[1:4]
req = urllib.request.Request(
    f"http://localhost:{port}/v1/models",
    headers={"Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.load(resp)
except Exception:
    sys.exit(1)
sys.exit(0 if name in [m.get("id") for m in data.get("data", [])] else 1)
PY
    then
      echo "endpoint ready: $endpoint"
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
  echo "error: the endpoint did not become ready within ${SERVER_TIMEOUT}s" >&2
  return 1
}
wait_for_server

# -- phase 2: generate every cell ------------------------------------------
export OPENAI_BASE_URL="$endpoint"
export OPENAI_API_KEY="$API_KEY"

declare -a paused=()
declare -a errored=()
for task in "${TASKS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    gen_args=(--backend openai_compatible --model "$SERVED_NAME"
              --task "$task" --condition "$condition"
              --chains "$CHAINS" --hops "$HOPS"
              --max-retries "$MAX_RETRIES"
              --partial-flush-every "$PARTIAL_FLUSH_EVERY"
              --workers "$NWORKERS")
    [[ -n "$SEEDS_FILE" ]] && gen_args+=(--seeds-file "$SEEDS_FILE")
    [[ -n "$LOGS_ROOT" ]]  && gen_args+=(--logs-root "$LOGS_ROOT")
    [[ -n "$CACHE_ROOT" ]] && gen_args+=(--cache-root "$CACHE_ROOT")
    if [[ -n "$TEMPERATURE" ]]; then
      gen_args+=(--temperature "$TEMPERATURE" --top-p "$TOP_P")
    fi
    if (( RERUN_FAILED )); then
      gen_args+=(--rerun-failed)
    else
      gen_args+=(--replicate 0)
    fi
    echo "=== [$task / $condition] $(date '+%Y-%m-%d %H:%M:%S') ==="
    rc=0
    "$PYTHON" "$GENERATOR" "${gen_args[@]}" || rc=$?
    if (( rc == 0 )); then
      continue
    elif (( rc == 3 )); then
      echo "cell paused: $task / $condition (re-run to resume from the cache)" >&2
      paused+=("$task/$condition")
    else
      echo "error: generator exited $rc for $task / $condition" >&2
      errored+=("$task/$condition (exit $rc)")
    fi
    if (( STOP_ON_PAUSE )); then
      echo "--stop-on-pause: stopping the cell loop" >&2
      break 2
    fi
  done
done

echo "=== summary ==="
echo "cells run:   $(( ${#TASKS[@]} * ${#CONDITIONS[@]} ))"
if (( ${#paused[@]} )); then
  echo "paused:      ${paused[*]}"
fi
if (( ${#errored[@]} )); then
  echo "errored:     ${errored[*]}"
fi
if (( ${#errored[@]} )); then
  exit 1
elif (( ${#paused[@]} )); then
  exit 3
fi
exit 0
