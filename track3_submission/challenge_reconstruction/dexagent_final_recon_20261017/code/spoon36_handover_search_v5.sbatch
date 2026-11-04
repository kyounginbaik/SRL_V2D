#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=14G
#SBATCH --time=00:45:00
#SBATCH --array=0-1
#SBATCH --job-name=spoon36-contact-cones
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-contact-cones-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then pose_name=pose_00; else pose_name=pose_01; fi
"$task_root/.venv/bin/python" "$task_root/code/physics_handover_search_v5.py" \
 --spec "$task_root/refinement_20261018/episode_000036/handover_mirrored_curled_inputs/$pose_name.json" \
 --warmstart "$task_root/refinement_20261018/episode_000036/handover_search_v4/$pose_name/result.json" \
 --output "$task_root/refinement_20261018/episode_000036/handover_search_v5/$pose_name" \
 --samples 128 --iterations 22 --workers 16 --seed "$((3650 + SLURM_ARRAY_TASK_ID))"
