#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-7%3
#SBATCH --job-name=expanded-physical-baseline
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/baseline-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd "$task_root"
episodes=(6 19 32 33 35 36 37 38)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python code/prepare_expanded_batch.py baseline --episodes "${episodes[$SLURM_ARRAY_TASK_ID]}"
