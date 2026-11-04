#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-2
#SBATCH --job-name=spoon36-learned
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-learned-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
case "$SLURM_ARRAY_TASK_ID" in
  0) gain=.01 ;;
  1) gain=.03 ;;
  2) gain=.1 ;;
esac
"$task_root/.venv/bin/python" "$task_root/code/train_episode36_feedback.py" \
 --teacher "$task_root/refinement_20261018/episode_000036/physical_release_search_v2" \
 --output "$task_root/refinement_20261018/episode_000036/learned_feedback_v2/gain_$gain" --gain "$gain"
