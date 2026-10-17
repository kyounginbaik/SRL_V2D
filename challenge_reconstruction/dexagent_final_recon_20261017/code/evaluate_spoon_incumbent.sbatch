#!/usr/bin/env bash
#SBATCH --job-name=spoon-full-replay
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/incumbent-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun "$task_root/.venv/bin/python" "$task_root/code/evaluate_pickup_incumbent.py" --search "$task_root/refinement_20261018/episode_000003/search_pose_00" --output "$task_root/refinement_20261018/episode_000003/first_incumbent_full"
