#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --job-name=cup32-real-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/cup32-pickup-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
pose=$(printf 'pose_%02d' "$SLURM_ARRAY_TASK_ID");spec=$(printf 'spec_%02d.json' "$SLURM_ARRAY_TASK_ID")
.venv/bin/python code/physics_pickup_search_v2.py --spec "refinement_20261018/expanded_batch/cup32_repaired_pickup_inputs/$spec" --output "refinement_20261018/expanded_batch/cup32_repaired_pickup/$pose" --samples 96 --iterations 16 --workers 16 --seed "$((132+SLURM_ARRAY_TASK_ID))"
