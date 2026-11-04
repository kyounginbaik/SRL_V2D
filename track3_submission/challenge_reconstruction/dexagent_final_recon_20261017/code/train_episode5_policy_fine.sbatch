#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --array=0-8%6
#SBATCH --job-name=ep5-policy-gains
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-policy-fine-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
candidate_root="$task_root/refinement_20261018/calibrated_scene_candidates"
teacher="$candidate_root/retention5/start_2.6/execution"
gains=(0.008 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.027)
gain="${gains[$SLURM_ARRAY_TASK_ID]}"
out="$task_root/refinement_20261018/learned_episode5_fine/gain_${gain}"
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$task_root/.venv/bin/python" "$task_root/code/learned_feedback_policy.py" \
 --scene "$teacher/scene_physics.xml" --trajectory "$teacher/trajectory.npz" \
 --output "$out/training" --centers 512 --feedback-gain "$gain" --velocity-gain 0.0005
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$task_root/.venv/bin/python" "$task_root/code/evaluate_versioned_feedback.py" \
 --episode 5 --episode-dir "$candidate_root/episodes/episode_000005" \
 --scene "$teacher/scene_physics.xml" --checkpoint "$out/training/policy.npz" --output "$out/evaluation"
