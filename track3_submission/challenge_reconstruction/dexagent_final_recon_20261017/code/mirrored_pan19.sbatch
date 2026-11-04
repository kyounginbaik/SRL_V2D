#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --job-name=pan19-mirror-grasp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/pan19-mirror-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
spec=$(printf 'spec_%02d.json' "$SLURM_ARRAY_TASK_ID");pose=$(printf 'pose_%02d' "$SLURM_ARRAY_TASK_ID")
.venv/bin/python code/physics_pickup_search_v2.py --spec "refinement_20261018/expanded_batch/mirrored_pan19_inputs/$spec" --output "refinement_20261018/expanded_batch/mirrored_pan19/$pose" --samples 96 --iterations 16 --workers 16 --seed "$((172+SLURM_ARRAY_TASK_ID))"
