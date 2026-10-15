#!/bin/bash
#SBATCH --job-name=chord100m
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=40:00:00
set -euo pipefail
# Submit from repository root. Override partition/nodelist/output via sbatch flags.
REPO_ROOT=${SLURM_SUBMIT_DIR:?Submit this script from the cloned repository root}
GPU_ALLOCATION=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}}
GPU_ALLOCATION=${GPU_ALLOCATION%%,*}
GPU_DEVICE=$(nvidia-smi -i "$GPU_ALLOCATION" --query-gpu=uuid --format=csv,noheader | head -1)
python3 "$REPO_ROOT/robotic_grounding/uniretarget/launch.py" \
    --episode "${1:-1}" --seed "${2:-43}" --phase "${4:-full}" --gpu "$GPU_DEVICE" --upper "${3:-gaussian}"
