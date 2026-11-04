#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-3
#SBATCH --job-name=spoon36-transfer-smoke-v3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-transfer-smoke-v3-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
pose_name=$(printf 'pose_%02d' "$SLURM_ARRAY_TASK_ID")
"$task_root/.venv/bin/python" "$task_root/code/physics_handover_search_v3.py" \
 --spec "$task_root/refinement_20261018/episode_000036/handover_mirrored_curled_inputs/$pose_name.json" \
 --output "$task_root/refinement_20261018/episode_000036/handover_smoke_v3/$pose_name" \
 --samples 2 --iterations 0 --workers 1 --seed 361
