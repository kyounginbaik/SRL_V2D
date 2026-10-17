#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=ep3-posture-anatomy
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/posture-anatomy-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_root/.venv/bin/python" "$task_root/code/search_posture_policy_v3.py" --output "$task_root/refinement_20261018/learned_episode3_posture_anatomy" --samples 64 --iterations 20 --workers 16
