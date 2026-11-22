#!/bin/bash
#SBATCH --job-name=serve-generate-qwen3-32b
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                 # full node: both A100 (2 x 40 GB) for TP=2
#SBATCH --time=3-00:00:00            # QOS max; use e.g. 00:30:00 for a first test
#SBATCH --output=%x-%j.log
##SBATCH --account=pXXXXX            # optional, otherwise default project is used

# ---- adjust these ----------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/scripts/confirmatory/serve_model_and_generate.sh"   # your path
MODEL="Qwen/Qwen3-32B"                      # check the exact HF id
SERVED_NAME="Qwen3-32B"
PORT=8892
# ----------------------------------------------------------------------------

module purge
module load miniconda3/24.1.2
eval "$(conda shell.bash hook)"
conda activate $DATA/envs/vllm_env

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
  --tp 2 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-mem-util 0.97 \
  --temperature 0.7 \
  --top-p 0.8 \
  --rerun-failed-pass \
  --nworkers 32
