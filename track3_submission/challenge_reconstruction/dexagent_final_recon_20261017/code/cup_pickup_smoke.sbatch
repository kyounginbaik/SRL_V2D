#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --job-name=ep10-cup-smoke
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000010/cup-smoke-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$SLURM_ARRAY_TASK_ID" in
  0) task_side=left;;
  1) task_side=right;;
esac
"$task_root/.venv/bin/python" "$task_root/code/physics_cup_pickup_search.py" --spec "$task_root/refinement_20261018/episode_000010/calibrated_pickup_inputs/$task_side.json" --output "$task_root/refinement_20261018/episode_000010/calibrated_pickup_smoke/$task_side" --iterations 0 --workers 1
