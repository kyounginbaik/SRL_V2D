#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --array=0-1
#SBATCH --job-name=spoon36-slide-clear
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-slide-clear-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then distance=.028; else distance=.045; fi
"$task_root/.venv/bin/python" "$task_root/code/spoon36_slide_thumb_clear.py" --distance "$distance"
