#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --job-name=ep5-calibrated-retain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-cal-retain-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
candidate_root="$task_root/refinement_20261018/calibrated_scene_candidates"
starts=(2.6 4.5)
start="${starts[$SLURM_ARRAY_TASK_ID]}"
srun --ntasks=1 "$task_root/.venv/bin/python" "$task_root/code/refine_versioned_retention.py" \
 --source "$candidate_root/pickup5/candidate_00" \
 --episode-dir "$candidate_root/episodes/episode_000005" \
 --output "$candidate_root/retention5/start_${start}" \
 --start-s "$start" --stop-s 13.55 --samples 128 --iterations 18 --workers 24 --seed "$((71+SLURM_ARRAY_TASK_ID))"
