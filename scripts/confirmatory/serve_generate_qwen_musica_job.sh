#!/bin/bash
#SBATCH --job-name=serve-generate-qwen3-32b-musica
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # one H100 (~94 GB) fits the 32B at BF16 (TP=1)
#SBATCH --time=3-00:00:00            # QOS max; use e.g. 00:30:00 for a first test
#SBATCH --output=%x-%j.log
# MUSICA needs an explicit account (project specific). Submit with
# ./submit_musica.sh or sbatch --account=<name>; no account is hardcoded here.

# ---- adjust these ----------------------------------------------------------
# Slurm runs a spooled copy, so BASH_SOURCE points at /var/spool/...;
# search upward from the submission directory for the repo root.
submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT=""
_search="$submit_dir"
for _ in 1 2 3 4; do
  if [[ -x "$_search/scripts/confirmatory/serve_model_and_generate.sh" ]]; then
    REPO_ROOT="$_search"
    break
  fi
  _search="$(dirname "$_search")"
done
if [[ -z "$REPO_ROOT" ]]; then
  echo "error: cannot find the repo from $submit_dir" >&2
  echo "       submit from the repo root or from scripts/confirmatory/" >&2
  exit 1
fi
RUN_SCRIPT="$REPO_ROOT/scripts/confirmatory/serve_model_and_generate.sh"   # your path
MODEL="Qwen/Qwen3-32B"                      # check the exact HF id
SERVED_NAME="Qwen3-32B"
PORT=8892
# ----------------------------------------------------------------------------

# Miniforge3 is an EESSI extension: load EESSI first, then the conda module.
module --force purge
module load EESSI/2025.06
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
export VLLM_ENV="${VLLM_ENV:-$DATA/envs/vllm_env}"
conda activate "$VLLM_ENV" || {
  echo "error: cannot activate $VLLM_ENV (set VLLM_ENV to override)" >&2
  exit 1
}

# Keep the weights out of the small $HOME quota
export HF_HOME="${DATA:-$HOME}/hf_cache"

echo "Job:        $SLURM_JOB_ID on $(hostname)"
echo "Partition:  $SLURM_JOB_PARTITION  QOS: $SLURM_JOB_QOS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Endpoint:   http://localhost:$PORT/v1"
nvidia-smi -L

# Serve Qwen3-32B, then run all 12 task x condition cells against it.
# Decoding is passed explicitly because the short served name is not in the
# generator's VENDOR_DECODE table (Qwen3-32B card: temperature 0.7, top_p 0.8).
# --rerun-failed-pass: after the first pass, finish paused cells and retry
# the chains recorded as failed, all against the same running server.
exec "$RUN_SCRIPT" \
  --model "$MODEL" \
  --served-name "$SERVED_NAME" \
  --port "$PORT" \
  --tp 1 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-mem-util 0.97 \
  --temperature 0.7 \
  --top-p 0.8 \
  --rerun-failed-pass \
  --nworkers 32
