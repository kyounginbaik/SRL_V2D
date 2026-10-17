#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ep1-carry-validate
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep1-carry-validate-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
repair_root="$task_root/refinement_20261018/episode_000001/reconstruction_repair/patch_height094"
selected="$repair_root/learned_carry_refinement_v2"
precision="$repair_root/learned_carry_precision6"
task_mode="${1:?precision or crosshost}"
if [[ "$task_mode" == precision ]]; then
  "$task_root/.venv/bin/python" "$task_root/code/evaluate_command_precision.py" --episode 1 --decimals 6 --baseline-report "$selected/evaluation/result.json" --output "$precision"
  eval_dir="$precision/evaluation"
elif [[ "$task_mode" == crosshost ]]; then
  eval_dir="$precision/crosshost_$SLURMD_NODENAME"
  "$task_root/.venv/bin/python" "$task_root/code/evaluate_policy_component.py" --episode 1 --episode-dir "$repair_root/repaired_adapter" --scene "$selected/evaluation/execution/scene_physics.xml" --checkpoint "$precision/policy.npz" --output "$eval_dir"
  "$task_root/.venv/bin/python" "$task_root/code/audit_full_joint_limits.py" --run "$eval_dir"
elif [[ "$task_mode" == rawcrosshost ]]; then
  eval_dir="$selected/crosshost_$SLURMD_NODENAME"
  "$task_root/.venv/bin/python" "$task_root/code/evaluate_policy_component.py" --episode 1 --episode-dir "$repair_root/repaired_adapter" --scene "$selected/evaluation/execution/scene_physics.xml" --checkpoint "$selected/policy.npz" --output "$eval_dir"
  "$task_root/.venv/bin/python" "$task_root/code/audit_full_joint_limits.py" --run "$eval_dir"
else
  echo "Unknown mode: $task_mode" >&2
  exit 2
fi
"$task_root/.venv/bin/python" "$task_root/code/audit_object_support.py" "$eval_dir"
"$task_root/.venv/bin/python" "$task_root/code/audit_episode1_alignment_bias.py" --result "$eval_dir/result.json"
