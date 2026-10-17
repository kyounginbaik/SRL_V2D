#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=ep3-posture-head
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/posture-head-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
teacher="$task_root/refinement_20261018/episode_000003/unused_fingers_v2/reference_1.00_start3.2/execution"
out="$task_root/refinement_20261018/learned_episode3_posture_head"
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$task_root/.venv/bin/python" "$task_root/code/posture_feedback_policy.py" --base-checkpoint "$task_root/refinement_20261018/learned_episode3/gain_0.01/training/policy.npz" --scene "$teacher/scene_physics.xml" --trajectory "$teacher/trajectory.npz" --output "$out/training"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$task_root/.venv/bin/python" "$task_root/code/evaluate_posture_feedback.py" --episode 3 --scene "$teacher/scene_physics.xml" --checkpoint "$out/training/policy.npz" --output "$out/evaluation"
