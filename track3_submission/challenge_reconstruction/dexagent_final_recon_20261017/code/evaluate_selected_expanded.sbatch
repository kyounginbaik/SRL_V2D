#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --job-name=selected-learned-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/selected-audit-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(6 33)
.venv/bin/python code/evaluate_expanded_selected.py --episode "${episodes[$SLURM_ARRAY_TASK_ID]}"
