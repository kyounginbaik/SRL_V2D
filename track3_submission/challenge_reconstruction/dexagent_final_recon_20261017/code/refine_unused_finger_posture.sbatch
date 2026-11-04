#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --array=0-3
#SBATCH --job-name=ep3-finger-posture
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/posture-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
modes=(neutral neutral reference reference)
amounts=(0.5 1.0 0.5 1.0)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_root/.venv/bin/python" "$task_root/code/refine_unused_finger_posture.py" --mode "${modes[$SLURM_ARRAY_TASK_ID]}" --amount "${amounts[$SLURM_ARRAY_TASK_ID]}"
