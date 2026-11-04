#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --array=6-9
#SBATCH --job-name=ep3-command-precision
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep3-precision-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/evaluate_command_precision.py" --episode 3 --decimals "$SLURM_ARRAY_TASK_ID" --node "$SLURMD_NODENAME"
