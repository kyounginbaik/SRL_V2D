#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --job-name=ep5-calibrated-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-cal-pickup-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
printf -v task_index '%02d' "$SLURM_ARRAY_TASK_ID"
srun --ntasks=1 "$task_root/.venv/bin/python" "$task_root/code/physics_pickup_search_v2.py" \
 --spec "$task_root/refinement_20261018/calibrated_scene_candidates/pickup_inputs5/spec_${task_index}.json" \
 --output "$task_root/refinement_20261018/calibrated_scene_candidates/pickup5/candidate_${task_index}" \
 --samples 96 --iterations 12 --workers 24 --seed "$((61+SLURM_ARRAY_TASK_ID))"
