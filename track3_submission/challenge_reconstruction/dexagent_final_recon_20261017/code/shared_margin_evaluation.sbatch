#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --job-name=shared-margin-eval
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/shared-margin-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
case "$SLURM_ARRAY_TASK_ID" in
 0) episode=episode_000003;;
 1) episode=episode_000005;;
esac
run="$task_root/refinement_20261018/pinky_margin_v2/$episode"
"$task_root/.venv/bin/python" "$task_root/code/reevaluate_component.py" --baseline-report "$run/evaluation/result.json" --output "$task_root/refinement_20261018/shared_evaluator/$episode"
