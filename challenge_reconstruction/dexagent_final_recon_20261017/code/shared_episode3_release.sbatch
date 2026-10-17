#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=shared-release3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/shared-release3-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/reevaluate_component.py" \
  --baseline-report "$task_root/refinement_20261018/episode_000003/quant6_release_gentle_v5/evaluation/result.json" \
  --output "$task_root/refinement_20261018/shared_evaluator_v3/episode_000003/$SLURMD_NODENAME"
