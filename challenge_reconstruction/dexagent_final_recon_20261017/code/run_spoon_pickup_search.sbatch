#!/usr/bin/env bash
#SBATCH --job-name=spoon-physical-cem
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/search-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
printf -v pose '%02d' "$SLURM_ARRAY_TASK_ID"
srun "$task_root/.venv/bin/python" "$task_root/code/physics_pickup_search.py" --spec "$task_root/refinement_20261018/episode_000003/pickup_inputs/pose_$pose.json" --output "$task_root/refinement_20261018/episode_000003/search_pose_$pose" --samples 128 --iterations 24 --workers 16 --seed 44
