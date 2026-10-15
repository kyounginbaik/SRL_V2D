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
# Numeric Slurm allocations can be indices or NVIDIA device minor numbers.
# Fail on ambiguity unless this site's convention or allocated UUID is explicit.
GPU_DEVICE=$(python3 "$REPO_ROOT/robotic_grounding/uniretarget/gpu_allocation.py")
python3 "$REPO_ROOT/robotic_grounding/uniretarget/launch.py" \
    --episode "${1:-1}" --seed "${2:-43}" --phase "${4:-full}" --gpu "$GPU_DEVICE" --upper "${3:-gaussian}"
