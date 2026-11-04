#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-5%6
#SBATCH --job-name=dustpan38-release
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/dustpan-release-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
starts=(10.2 10.5 10.8)
openings=(0.008 0.012)
start=${starts[$((SLURM_ARRAY_TASK_ID / 2))]}
opening=${openings[$((SLURM_ARRAY_TASK_ID % 2))]}
.venv/bin/python code/repair_dustpan38_release.py --start-s "$start" --opening-m "$opening" \
 --output "refinement_20261018/expanded_batch/dustpan_release/episode_000038/case_${SLURM_ARRAY_TASK_ID}"
