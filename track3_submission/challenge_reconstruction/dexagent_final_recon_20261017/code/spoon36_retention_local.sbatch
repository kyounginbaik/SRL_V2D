#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=spoon36-retention
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-retention-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/refine_versioned_retention.py" \
 --source "$task_root/refinement_20261018/spoon36_long_grasp/search" \
 --episode-dir "$task_root/episodes/episode_000036" \
 --output "$task_root/refinement_20261018/episode_000036/retention_after9" \
 --start-s 9 --stop-s 18.0 --side right --protect-object wooden_spoon \
 --samples 128 --iterations 24 --workers 24 --seed 3619
