#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --array=0-2
#SBATCH --job-name=ep8-learned-policy
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep8-policy-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd "$task_root"
teacher="$task_root/refinement_20261017/feedback_polish/episode_000008/both_0.5/execution"
gains=(0.01 0.03 0.1)
gain="${gains[$SLURM_ARRAY_TASK_ID]}"
out="$task_root/refinement_20261018/learned_episode8/gain_${gain}"
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python code/learned_feedback_policy.py \
  --scene "$teacher/scene_physics.xml" --trajectory "$teacher/trajectory.npz" \
  --output "$out/training" --centers 512 --feedback-gain "$gain" --velocity-gain 0.0005
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python code/evaluate_learned_feedback.py \
  --episode 8 --scene "$teacher/scene_physics.xml" --checkpoint "$out/training/policy.npz" --output "$out/evaluation"
