#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=111-112
#SBATCH --job-name=ep3-precision-grasp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep3-precision-grasp-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$task_root/.venv/bin/python" "$task_root/code/search_quantized_posture.py" --output "$task_root/refinement_20261018/quantized_posture/seed_$SLURM_ARRAY_TASK_ID" --samples 128 --iterations 12 --workers 24 --seed "$SLURM_ARRAY_TASK_ID"
