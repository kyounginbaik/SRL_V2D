#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --job-name=ep10-blue-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000010/blue-pickup-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_root/.venv/bin/python" "$task_root/code/physics_cup_pickup_search.py" --spec "$task_root/refinement_20261018/episode_000010/calibrated_pickup_inputs/left.json" --output "$task_root/refinement_20261018/episode_000010/calibrated_blue_pickup" --samples 128 --iterations 20 --workers 24 --seed 43
