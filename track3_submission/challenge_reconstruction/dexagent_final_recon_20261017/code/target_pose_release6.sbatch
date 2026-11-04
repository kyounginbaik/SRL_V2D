#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --job-name=spoon6-place
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/spoon6-place-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
.venv/bin/python code/fit_target_pose_release6.py --case "$SLURM_ARRAY_TASK_ID"
