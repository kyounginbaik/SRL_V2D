#!/bin/bash
#SBATCH --job-name=ep6-target-margin
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --array=0-7%8
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/logs/target-margin6-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
.venv/bin/python code/test_policy_target_margins.py --case "$SLURM_ARRAY_TASK_ID"
