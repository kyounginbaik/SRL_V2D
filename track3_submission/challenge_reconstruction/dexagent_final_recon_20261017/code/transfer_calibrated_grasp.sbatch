#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --array=0-15%16
#SBATCH --job-name=ep5-calibrated-spoon
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-cal-spoon-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun --ntasks=1 "$task_root/.venv/bin/python" "$task_root/code/transfer_calibrated_grasp.py" --index "$SLURM_ARRAY_TASK_ID"
