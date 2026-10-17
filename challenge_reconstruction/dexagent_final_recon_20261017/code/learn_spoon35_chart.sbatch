#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --array=0-2
#SBATCH --job-name=learn-spoon35-chart
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/learn-spoon35-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
teacher="$task_root/refinement_20261018/spoon_local_retention/episode_000035"
adapter="$task_root/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter"
gains=(0.01 0.03 0.1)
out="$task_root/refinement_20261018/learned_spoon35_chart/gain_${gains[$SLURM_ARRAY_TASK_ID]}"
"$task_root/.venv/bin/python" "$task_root/code/learned_feedback_policy.py" \
  --scene "$teacher/execution/scene_physics.xml" --trajectory "$teacher/execution/trajectory.npz" \
  --output "$out/training" --centers 1024 --feedback-gain "${gains[$SLURM_ARRAY_TASK_ID]}"
"$task_root/.venv/bin/python" "$task_root/code/evaluate_policy_component.py" \
  --episode 35 --episode-dir "$adapter" --scene "$teacher/execution/scene_physics.xml" \
  --checkpoint "$out/training/policy.npz" --output "$out/evaluation"
"$task_root/.venv/bin/python" "$task_root/code/audit_full_joint_limits.py" --run "$out/evaluation"
