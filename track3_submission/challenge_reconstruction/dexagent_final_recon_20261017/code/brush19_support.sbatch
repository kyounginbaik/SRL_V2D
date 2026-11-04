#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --array=0-2
#SBATCH --job-name=brush19-support
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/brush19-support-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
alphas=(0.5 0.75 1.0)
.venv/bin/python code/audit_brush19_bristle_support.py --alpha "${alphas[$SLURM_ARRAY_TASK_ID]}"
