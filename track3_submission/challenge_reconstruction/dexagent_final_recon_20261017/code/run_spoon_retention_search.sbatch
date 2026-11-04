#!/usr/bin/env bash
#SBATCH --job-name=spoon-retain-cem
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/retention-search-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun "$task_root/.venv/bin/python" "$task_root/code/physics_retention_search.py" --source "$task_root/refinement_20261018/episode_000003/search_pose_00" --output "$task_root/refinement_20261018/episode_000003/retention_after45" --samples 128 --iterations 20 --workers 32 --seed 45
