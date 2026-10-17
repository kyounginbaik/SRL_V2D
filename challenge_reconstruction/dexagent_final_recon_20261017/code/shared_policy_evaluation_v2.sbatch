#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --job-name=shared-policy-eval
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/shared-eval-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
case "$SLURM_ARRAY_TASK_ID" in
 0) episode=episode_000008; run="$task_root/refinement_20261018/learned_episode8/gain_0.01";;
 1) episode=episode_000037; run="$task_root/refinement_20261018/expanded_batch/learned/episode_000037/gain_0.03";;
esac
"$task_root/.venv/bin/python" "$task_root/code/reevaluate_component.py" --baseline-report "$run/evaluation/result.json" --output "$task_root/refinement_20261018/shared_evaluator_v2/$episode"
