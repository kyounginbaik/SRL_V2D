#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-14%15
#SBATCH --job-name=expanded-wrist-smooth
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/wrist-smooth-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
episodes=(35 36 6)
anchors=(2.8 2.8 3.5)
sigmas=(0.5 1 2 3 4)
episode=$(printf 'episode_%06d' "${episodes[$((SLURM_ARRAY_TASK_ID / 5))]}")
anchor=${anchors[$((SLURM_ARRAY_TASK_ID / 5))]}
sigma=${sigmas[$((SLURM_ARRAY_TASK_ID % 5))]}
.venv/bin/python code/smooth_pickup_wrist.py \
 --source "refinement_20261018/expanded_batch/pickup_search/${episode}/anchor_${anchor}" \
 --output "refinement_20261018/expanded_batch/wrist_smoothing/${episode}/sigma_${sigma}" \
 --sigma "$sigma" --start-s 2.5
