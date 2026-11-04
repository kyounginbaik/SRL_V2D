#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --job-name=spoon36-release-CEM
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-release-CEM-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/physics_spoon36_release_search.py" --output "$task_root/refinement_20261018/episode_000036/physical_release_search" --samples 128 --iterations 25 --workers 16 --seed 3660
