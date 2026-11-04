#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ep3-hold-release
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/hold-release-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$task_root/.venv/bin/python" "$task_root/code/run_posture_hold_release.py"
