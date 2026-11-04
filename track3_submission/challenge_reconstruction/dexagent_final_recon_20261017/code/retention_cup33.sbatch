#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --array=0-2%3
#SBATCH --job-name=cup33-retention
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/cup33-retention-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
seed=$((70 + SLURM_ARRAY_TASK_ID))
.venv/bin/python code/refine_versioned_retention.py \
 --source refinement_20261018/expanded_batch/cup33_pickup/pose_03 --episode-dir episodes/episode_000033 \
 --output "refinement_20261018/expanded_batch/cup33_retention/seed_${seed}" \
 --start-s 5.8 --stop-s 16.25 --protect-object blue_cup --samples 96 --iterations 16 --workers 16 --seed "$seed"
