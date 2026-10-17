#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --job-name=expanded-hand-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/hand-margins-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
episodes=(6 33 38)
variants=(basis_1024_gain_0.1 basis_1024_gain_0.01 basis_1024_gain_0.1)
ep=${episodes[$SLURM_ARRAY_TASK_ID]};name=$(printf 'episode_%06d' "$ep")
run="refinement_20261018/expanded_batch/learned_fine/${name}/${variants[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python code/search_hand_margins.py --checkpoint "$run/training/policy.npz" \
 --scene "$run/evaluation/execution/scene_physics.xml" --episode-dir "episodes/$name" \
 --baseline-report "$run/evaluation/result.json" --config "refinement_20261018/expanded_batch/hand_margin_inputs/${name}.json" \
 --output "refinement_20261018/expanded_batch/hand_margin_policy/${name}" --samples 96 --iterations 16 --workers 16 --seed "$((93 + SLURM_ARRAY_TASK_ID))"
