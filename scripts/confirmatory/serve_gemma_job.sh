#!/bin/bash
#SBATCH --job-name=serve-gemma-4-31b
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                 # full node: both A100 (2 x 40 GB) for TP=2
#SBATCH --time=3-00:00:00            # QOS max; use e.g. 00:30:00 for a first test
#SBATCH --output=%x-%j.log
##SBATCH --account=pXXXXX            # optional, otherwise default project is used

# ---- adjust these ----------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR}"                    # submit from the repo root
SERVE_SCRIPT="$REPO_ROOT/scripts/confirmatory/serve_model.sh"   # your path
MODEL="google/gemma-4-31B-it"                      # check the exact HF id
PORT=8892
# huggingface api keys here?
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
echo "Endpoint:   http://$(hostname):$PORT/v1"
nvidia-smi -L

"$SERVE_SCRIPT" \
  --model "$MODEL" \
  --served-name gemma-4-31b-it \
  --port "$PORT" \
  --tp 2 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-mem-util 0.92
