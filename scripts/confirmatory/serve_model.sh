#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one model behind an OpenAI-compatible endpoint (vLLM).
#
# One script for every environment: laptop CPU, local GPU workstation,
# ASC-5 cluster. Only the parameters differ. CPU serving needs a vLLM
# build with CPU support; pass --device cpu so vLLM does not try to
# auto-detect a GPU.
#
#   Laptop, tiny model:   --model HuggingFaceTB/SmolLM2-135M-Instruct --device cpu --dtype float32
#   4080, Llama 8-bit:    --model meta-llama/Llama-3.1-8B-Instruct --quant bitsandbytes
#   ASC-5, Llama BF16:    --model meta-llama/Llama-3.1-8B-Instruct --tp 1 --slurm
#   ASC-5, Qwen/Gemma:    --model Qwen/Qwen3-32B --tp 2 --slurm
#
# The client side is always the frozen generator:
#   OPENAI_BASE_URL=http://localhost:$PORT/v1 \
#   OPENAI_API_KEY=local \
#   python scripts/confirmatory/generate_chains_confirmatory.py \
#       --backend openai_compatible --model "$SERVED_NAME" ...
#
# Defaults can come from environment variables; command-line flags win.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-}"
SERVED_NAME="${SERVED_NAME:-}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DTYPE="${DTYPE:-}"
DEVICE="${DEVICE:-}"
QUANT="${QUANT:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
API_KEY="${API_KEY:-local}"
SLURM="${SLURM:-0}"
PARTITION="${PARTITION:-zen3_0512_a100x2}"
TIME="${TIME:-3-00:00:00}"
JOB_NAME="${JOB_NAME:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CHECK=0

usage() {
  cat <<'EOF'
Usage: serve_model.sh --model MODEL [options]

Options:
  --model MODEL          HF id or local path (required)
  --served-name NAME     name the API exposes (default: MODEL)
  --port PORT            HTTP port (default 8000)
  --tp N                 tensor-parallel size (default 1)
  --dtype DTYPE          float32 | float16 | bfloat16 (default: vLLM auto)
  --device DEVICE        cpu | cuda (default: vLLM auto-detect;
                         CPU-only machines must pass --device cpu)
  --quant QUANT          e.g. bitsandbytes (default: none)
  --max-model-len N      context cap (default: model config)
  --gpu-mem-util F       vLLM default 0.9; lower on shared cards
  --extra-args "... "    further vllm serve args
  --api-key KEY          value clients send (default: local)
  --slurm                submit as an sbatch job instead of running inline
  --partition NAME       Slurm partition (default zen3_0512_a100x2)
  --time TIME            Slurm walltime (default 3-00:00:00, partition max)
  --job-name NAME        Slurm job name (default serve-<served name>)
  --check                print the command / sbatch script, do not run
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
    --device)        DEVICE="$2"; shift 2 ;;
    --quant)         QUANT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2 ;;
    --extra-args)    EXTRA_ARGS="$2"; shift 2 ;;
    --api-key)       API_KEY="$2"; shift 2 ;;
    --slurm)         SLURM=1; shift ;;
    --partition)     PARTITION="$2"; shift 2 ;;
    --time)          TIME="$2"; shift 2 ;;
    --job-name)      JOB_NAME="$2"; shift 2 ;;
    --check)         CHECK=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage 1; }
[[ -n "$SERVED_NAME" ]] || SERVED_NAME="$MODEL"
[[ -n "$JOB_NAME" ]] || JOB_NAME="serve-$(basename "$SERVED_NAME" \
  | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-\{1,\}$//')"

SERVE=("$VLLM_BIN" serve "$MODEL"
       --served-model-name "$SERVED_NAME"
       --host 0.0.0.0 --port "$PORT"
       --tensor-parallel-size "$TP"
       --api-key "$API_KEY")
[[ -n "$DEVICE" ]]        && SERVE+=(--device "$DEVICE")
[[ -n "$DTYPE" ]]         && SERVE+=(--dtype "$DTYPE")
[[ -n "$QUANT" ]]         && SERVE+=(--quantization "$QUANT")
[[ -n "$MAX_MODEL_LEN" ]] && SERVE+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$GPU_MEM_UTIL" ]]  && SERVE+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
[[ -n "$EXTRA_ARGS" ]]    && SERVE+=($EXTRA_ARGS)

render_cmd() { printf '%q ' "${SERVE[@]}"; }

slurm_script() {
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=%s\n' "$JOB_NAME"
  printf '#SBATCH --partition=%s\n' "$PARTITION"
  printf '#SBATCH --gres=gpu:%s\n' "$TP"
  printf '#SBATCH --time=%s\n' "$TIME"
  printf '#SBATCH --output=%s-%%j.log\n\n' "$JOB_NAME"
  render_cmd; printf '\n'
}

echo "model:        $MODEL"
echo "served name:  $SERVED_NAME"
echo "endpoint:     http://0.0.0.0:$PORT/v1"
echo "device/dtype: ${DEVICE:-auto} / ${DTYPE:-auto}${QUANT:+ (quant=$QUANT)}"
[[ "$SLURM" == "1" ]] && echo "slurm:        partition=$PARTITION time=$TIME gres=gpu:$TP"

if [[ "$CHECK" == "1" ]]; then
  if [[ "$SLURM" == "1" ]]; then
    echo "--- sbatch script (not submitted) ---"
    slurm_script
  else
    echo "--- command (not run) ---"
    render_cmd; echo
  fi
  exit 0
fi

if [[ "$SLURM" == "1" ]]; then
  command -v sbatch >/dev/null || { echo "error: sbatch not found" >&2; exit 1; }
  tmp="$(mktemp "${TMPDIR:-/tmp}/serve_model_XXXXXX.sbatch")"
  slurm_script > "$tmp"
  echo "submitting: $tmp"
  sbatch "$tmp"
else
  command -v "$VLLM_BIN" >/dev/null || {
    echo "error: '$VLLM_BIN' not on PATH (pip install vllm, or set VLLM_BIN)" >&2
    exit 1
  }
  exec "${SERVE[@]}"
fi
