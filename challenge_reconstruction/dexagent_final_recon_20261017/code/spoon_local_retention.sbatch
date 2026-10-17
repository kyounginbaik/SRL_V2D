#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --job-name=spoon-local-hold
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon-local-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  source_run="$task_root/refinement_20261018/wrist_chart/episode_000035/chart_0"
  adapter="$source_run/adapter"
  start=5.5
  stop=19.0
  ep=35
else
  source_run="$task_root/refinement_20261018/expanded_batch/retention/episode_000036"
  adapter="$task_root/episodes/episode_000036"
  start=3.5
  stop=17.0
  ep=36
fi
"$task_root/.venv/bin/python" "$task_root/code/refine_versioned_retention.py" \
  --source "$source_run" --episode-dir "$adapter" \
  --output "$task_root/refinement_20261018/spoon_local_retention/episode_0000$ep" \
  --start-s "$start" --stop-s "$stop" --side right --protect-object wooden_spoon \
  --samples 64 --iterations 12 --workers 16 --seed "$((3500+ep))"
