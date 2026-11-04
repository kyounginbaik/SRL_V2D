#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --array=0-2
#SBATCH --job-name=cup33-release
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/cup33-release-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
starts=(15.4 15.6 15.8)
.venv/bin/python code/fit_expanded_supported_release.py --episode 33 --start "${starts[$SLURM_ARRAY_TASK_ID]}" --opening .008
