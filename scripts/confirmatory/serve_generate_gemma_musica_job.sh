#!/bin/bash
#SBATCH --job-name=serve-generate-gemma-4-31b-musica
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # one H100 (~94 GB) fits the 31B at BF16 (TP=1)
#SBATCH --time=3-00:00:00            # QOS max; use e.g. 00:30:00 for a first test
#SBATCH --output=%x-%j.log
# MUSICA needs an explicit account (project specific). Submit with
# ./submit_musica.sh or sbatch --account=<name>; no account is hardcoded here.

# ---- adjust these ----------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/scripts/confirmatory/serve_model_and_generate.sh"   # your path
MODEL="google/gemma-4-31B-it"                      # check the exact HF id
SERVED_NAME="gemma-4-31b-it"
PORT=8892
# ----------------------------------------------------------------------------

module purge
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/envs/vllm_env

# Keep the weights out of the small $HOME quota
export HF_HOME="${DATA:-$HOME}/hf_cache"

echo "Job:        $SLURM_JOB_ID on $(hostname)"
echo "Partition:  $SLURM_JOB_PARTITION  QOS: $SLURM_JOB_QOS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Endpoint:   http://localhost:$PORT/v1"
nvidia-smi -L

# Serve gemma-4-31B-it, then run all 12 task x condition cells against it.
# Decoding is passed explicitly because the short served name is not in the
# generator's VENDOR_DECODE table (Gemma 4 card: temperature 1.0, top_p 0.95).
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
  --temperature 1.0 \
  --top-p 0.95 \
  --rerun-failed-pass \
  --nworkers 32
