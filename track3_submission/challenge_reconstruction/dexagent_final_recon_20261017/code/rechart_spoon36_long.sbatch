#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --job-name=spoon36-long-chart
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-chart-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/rechart_floating_wrist.py" --episode 36 \
 --chart "$SLURM_ARRAY_TASK_ID" --source "$task_root/refinement_20261018/spoon36_long_grasp/search" \
 --output "$task_root/refinement_20261018/spoon36_long_chart"
