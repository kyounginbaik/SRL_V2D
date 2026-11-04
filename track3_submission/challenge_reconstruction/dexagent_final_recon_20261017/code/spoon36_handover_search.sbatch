#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=14G
#SBATCH --time=00:45:00
#SBATCH --array=0-1
#SBATCH --job-name=spoon36-transfer-CEM
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-transfer-CEM-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then pose_name=pose_03; else pose_name=pose_00; fi
"$task_root/.venv/bin/python" "$task_root/code/physics_handover_search.py" \
 --spec "$task_root/refinement_20261018/episode_000036/handover_inputs/$pose_name.json" \
 --output "$task_root/refinement_20261018/episode_000036/handover_search/$pose_name" \
 --samples 128 --iterations 28 --workers 16 --seed "$((3610 + SLURM_ARRAY_TASK_ID))"
