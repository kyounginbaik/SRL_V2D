#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --job-name=learned-pinky-v2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/pinky-v2-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
module=learned_feedback_policy
policy_class=LearnedPolicy
hands=both
extra=()
case "$SLURM_ARRAY_TASK_ID" in
 0) episode=3
    module=posture_feedback_policy_v4
    policy_class=PosturePolicy
    hands=left
    run="$task_root/refinement_20261018/learned_episode3_posture_grasp"
    checkpoint="$run/policy.npz"
    input_dir="$task_root/episodes/episode_000003" ;;
 1) episode=5
    run="$task_root/refinement_20261018/learned_episode5_fine/gain_0.022"
    checkpoint="$run/training/policy.npz"
    input_dir="$task_root/refinement_20261018/calibrated_scene_candidates/episodes/episode_000005" ;;
 2) episode=8
    extra=(--right-startup-only)
    run="$task_root/refinement_20261018/learned_episode8/gain_0.01"
    checkpoint="$run/training/policy.npz"
    input_dir="$task_root/episodes/episode_000008" ;;
esac
printf -v name 'episode_%06d' "$episode"
srun --ntasks=1 "$task_root/.venv/bin/python" "$task_root/code/search_pinky_margins_v2.py" \
 --checkpoint "$checkpoint" --base-module "$module" --base-class "$policy_class" \
 --scene "$run/evaluation/execution/scene_physics.xml" --episode-dir "$input_dir" \
 --baseline-report "$run/evaluation/result.json" --hands "$hands" \
 --output "$task_root/refinement_20261018/pinky_margin_v2/$name" \
 --samples 128 --iterations 12 --workers 24 --seed "$((102+SLURM_ARRAY_TASK_ID))" \
 --warm-start "$task_root/refinement_20261018/pinky_margin_v2_inputs/$episode.json" "${extra[@]}"
