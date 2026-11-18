#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one model behind an OpenAI-compatible endpoint (vLLM).
#
# One script for every environment: local GPU workstation and the ASC-5
# cluster. Only the parameters differ. When no GPU is visible the
# script selects cpu automatically (vLLM cannot infer a device on a
# CPU-only machine); --device overrides the detection. The device is
# exported as VLLM_TARGET_DEVICE because vLLM resolves its platform at
# import time, before any CLI parsing. CPU serving needs a vLLM build
# with CPU support.
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

# Load the repo's .env (HF_TOKEN etc.) when present; real env vars win.
# NOTE: resolve repo_root from the absolute script path, not BASH_SOURCE,
# which is relative when the script is invoked as ./serve_model.sh.
script_abs="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_abs")/../.." && pwd)"
if [[ -z "${HF_TOKEN:-}" && -f "$repo_root/.env" ]]; then
  # Read only HF_TOKEN from .env so a provider API_KEY does not become the
  # local OpenAI-compatible server key.
  HF_TOKEN="$(python3 - "$repo_root/.env" <<'PY'
from pathlib import Path
import sys
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if line.startswith("HF_TOKEN="):
        v = line.split("=", 1)[1].strip().strip('"').strip("'")
        print(v)
        break
PY
)"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN not found. Checked $repo_root/.env" >&2
  echo "       Add HF_TOKEN=... there or export HF_TOKEN before running." >&2
  exit 1
fi
# vLLM/transformers read HF_HUB_TOKEN (older builds only HF_TOKEN); set both.
export HF_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN="$HF_TOKEN"
echo "HF token loaded from ${repo_root}/.env" >&2

MODEL="${MODEL:-}"
SERVED_NAME="${SERVED_NAME:-}"
PORT="${PORT:-8892}"
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
VLLM_PY="${VLLM_PY:-}"
VLLM_USE_MODULE="${VLLM_USE_MODULE:-}"
HF_SNAPSHOT="${HF_SNAPSHOT:-auto}"
CHECK=0

usage() {
  cat <<'EOF'
Usage: serve_model.sh --model MODEL [options]

Options:
  --model MODEL          HF id or local path (required)
  --served-name NAME     name the API exposes (default: MODEL)
  --port PORT            HTTP port (default 8892)
  --tp N                 tensor-parallel size (default 1)
  --dtype DTYPE          float32 | float16 | bfloat16 (default: vLLM auto)
  --device DEVICE        cpu | cuda (default: cuda if a GPU is visible,
                         otherwise cpu); exported as VLLM_TARGET_DEVICE
                         because vllm serve has no --device flag
  --python PY            python interpreter to use with --module (default:
                         the python next to $VLLM_BIN, or python3)
  --module               run vLLM via 'python -m vllm.entrypoints.cli.main'
                         (auto-enabled when the 'vllm' console script cannot
                         find vllm package metadata, as with vllm-cpu wheels)
  --hf-snapshot MODE     auto | never | always. auto/always predownload HF
                         model ids to the local cache and serve that snapshot
                         path, while keeping --served-model-name as MODEL.
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
    --python)        VLLM_PY="$2"; shift 2 ;;
    --module)        VLLM_USE_MODULE=1; shift ;;
    --no-module)     VLLM_USE_MODULE=0; shift ;;
    --hf-snapshot)   HF_SNAPSHOT="$2"; shift 2 ;;
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
if [[ -z "$DEVICE" && "$SLURM" != "1" ]]; then
  # vLLM raises "Failed to infer device type" when no GPU is present.
  # sbatch jobs detect inside the allocation (the submit host may be GPU-less).
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE=""
  else
    DEVICE="cpu"
  fi
fi
[[ -n "$SERVED_NAME" ]] || SERVED_NAME="$MODEL"
[[ -n "$JOB_NAME" ]] || JOB_NAME="serve-$(basename "$SERVED_NAME" \
  | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-\{1,\}$//')"

MODEL_PATH="$MODEL"
if [[ "$CHECK" != "1" && "$HF_SNAPSHOT" != "never" && "$MODEL" != /* && "$MODEL" == */* ]]; then
  # Work around vLLM/Transformers paths that do not forward --hf-token to all
  # internal config lookups for gated repos: download with huggingface_hub
  # first (using HF_TOKEN/HF_HUB_TOKEN), then serve the local snapshot path.
  resolve_vllm_py_for_snapshot() {
    if [[ -n "$VLLM_PY" ]]; then echo "$VLLM_PY"; return; fi
    local bin; bin="$(command -v "$VLLM_BIN" 2>/dev/null || true)"
    if [[ -n "$bin" && -x "$(dirname "$bin")/python" ]]; then
      echo "$(dirname "$bin")/python"
    else
      echo "python3"
    fi
  }
  _py="$(resolve_vllm_py_for_snapshot)"
  echo "resolving HF snapshot for $MODEL with $_py ..." >&2
  MODEL_PATH="$($_py - "$MODEL" "$repo_root/.env" <<'PY'
import os, sys, tempfile
from pathlib import Path
repo = sys.argv[1]
env_path = Path(sys.argv[2])
tok = None
for raw in env_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if line.startswith("HF_TOKEN="):
        tok = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
if not tok:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
if not tok:
    raise SystemExit(f"HF token missing: checked {env_path}")
# Isolate auth lookup from any cached/expired login token (e.g. old whisperx).
home = tempfile.mkdtemp(prefix="hf_auth_")
os.environ["HF_HOME"] = home
os.environ["HF_TOKEN"] = tok
os.environ["HF_HUB_TOKEN"] = tok
from huggingface_hub import HfApi, snapshot_download
HfApi(token=tok).model_info(repo)
print(snapshot_download(repo_id=repo, token=tok, local_files_only=False))
PY
)"
  if [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]]; then
    export HF_HUB_OFFLINE=1
  fi
fi


resolve_vllm_py() {
  if [[ -z "$VLLM_PY" ]]; then
    local bin; bin="$(command -v "$VLLM_BIN" 2>/dev/null || true)"
    if [[ -n "$bin" && -x "$(dirname "$bin")/python" ]]; then
      VLLM_PY="$(dirname "$bin")/python"
    else
      VLLM_PY="python3"
    fi
  fi
}

# vllm-cpu wheels expose metadata as 'vllm-cpu', so the 'vllm' console script
# crashes at startup; fall back to 'python -m vllm.entrypoints.cli.main'.
if [[ -z "$VLLM_USE_MODULE" ]]; then
  if [[ "$CHECK" != "1" ]]; then
    command -v "$VLLM_BIN" >/dev/null 2>&1 || {
      echo "error: '$VLLM_BIN' not on PATH (activate the env or set VLLM_BIN)" >&2
      exit 1
    }
  fi
  resolve_vllm_py
  if "$VLLM_PY" -c "import importlib.metadata as m; m.version('vllm')"         >/dev/null 2>&1; then
    VLLM_USE_MODULE=0
  else
    VLLM_USE_MODULE=1
    echo "note: 'vllm' metadata missing; using $VLLM_PY -m vllm.entrypoints.cli.main" >&2
  fi
fi

# vllm-cpu wheels ship metadata as 'vllm-cpu', but vllm's CLI calls
# importlib.metadata.version("vllm") and crashes. Patch metadata.version
# to fall back to 'vllm-cpu' before invoking the CLI.
VLLM_CPU_PATCH='import importlib.metadata as m,sys
_o=m.version
def _v(n):
    try: return _o(n)
    except m.PackageNotFoundError:
        if n=="vllm":
            try: return _o("vllm-cpu")
            except Exception: return "0.0.0"
        raise
m.version=_v
sys.argv=["vllm"]+sys.argv[1:]
from vllm.entrypoints.cli import main as _main
_main.main()'

if [[ "$VLLM_USE_MODULE" == "1" ]]; then
  resolve_vllm_py
  SERVE=("$VLLM_PY" -c "$VLLM_CPU_PATCH" serve "$MODEL_PATH"
         --served-model-name "$SERVED_NAME"
         --host 0.0.0.0 --port "$PORT"
         --tensor-parallel-size "$TP"
         --api-key "$API_KEY")
else
  SERVE=("$VLLM_BIN" serve "$MODEL_PATH"
         --served-model-name "$SERVED_NAME"
         --host 0.0.0.0 --port "$PORT"
         --tensor-parallel-size "$TP"
         --api-key "$API_KEY")
fi
[[ -n "${HF_TOKEN:-}" ]]  && SERVE+=(--hf-token "$HF_TOKEN")
[[ -n "$DTYPE" ]]         && SERVE+=(--dtype "$DTYPE")
[[ -n "$QUANT" ]]         && SERVE+=(--quantization "$QUANT")
[[ -n "$MAX_MODEL_LEN" ]] && SERVE+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$GPU_MEM_UTIL" ]]  && SERVE+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
[[ -n "$EXTRA_ARGS" ]]    && SERVE+=($EXTRA_ARGS)

render_cmd() {
  local mask_next=0
  for arg in "${SERVE[@]}"; do
    if [[ "$mask_next" == "1" ]]; then
      printf '<set> '
      mask_next=0
      continue
    fi
    case "$arg" in
      --api-key|--hf-token) printf '%q ' "$arg"; mask_next=1 ;;
      *) printf '%q ' "$arg" ;;
    esac
  done
}

slurm_script() {
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=%s\n' "$JOB_NAME"
  printf '#SBATCH --partition=%s\n' "$PARTITION"
  printf '#SBATCH --gres=gpu:%s\n' "$TP"
  printf '#SBATCH --time=%s\n' "$TIME"
  printf '#SBATCH --output=%s-%%j.log\n\n' "$JOB_NAME"
  [[ -n "$DEVICE" ]] && printf 'export VLLM_TARGET_DEVICE=%s\n' "$DEVICE"
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
  [[ -n "$DEVICE" ]] && export VLLM_TARGET_DEVICE="$DEVICE"
  if [[ "$DEVICE" == "cpu" ]]; then
    echo "note: CPU serving needs a vLLM build with CPU support" >&2
    echo "      (vllm-cpu plus a CPU Triton build). GPU hosts do not" >&2
    echo "      need this." >&2
  fi
  export -n VLLM_BIN VLLM_PY VLLM_USE_MODULE 2>/dev/null || true
  if [[ "$VLLM_USE_MODULE" == "1" ]]; then
    command -v "$VLLM_PY" >/dev/null || {
      echo "error: '$VLLM_PY' not on PATH (set --python)" >&2; exit 1; }
  else
    command -v "$VLLM_BIN" >/dev/null || {
      echo "error: '$VLLM_BIN' not on PATH (pip install vllm, or set VLLM_BIN)" >&2
      exit 1; }
  fi
  exec "${SERVE[@]}"
fi
