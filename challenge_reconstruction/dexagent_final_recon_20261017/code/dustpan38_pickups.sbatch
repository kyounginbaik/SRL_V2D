#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%6
#SBATCH --job-name=dustpan38-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/dustpan-pickup-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case_id=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
.venv/bin/python code/physics_pickup_search_v2.py \
 --spec "refinement_20261018/expanded_batch/dustpan_geometry/episode_000038/spec_${case_id}.json" \
 --output "refinement_20261018/expanded_batch/dustpan_pickup/episode_000038/pose_${case_id}" \
 --samples 64 --iterations 16 --workers 12 --seed 44
