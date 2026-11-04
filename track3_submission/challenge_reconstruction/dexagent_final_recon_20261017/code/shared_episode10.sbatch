#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=shared-policy10
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/shared-policy10-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/reevaluate_component.py" \
  --baseline-report "$task_root/refinement_20261018/episode_000010/learned_feedback/gain_0.03/evaluation/result.json" \
  --output "$task_root/refinement_20261018/shared_evaluator_v2/episode_000010"
