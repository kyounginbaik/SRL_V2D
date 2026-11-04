#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --job-name=spoon36-pinky-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-pinky-margins-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
task_actor="$task_root/refinement_20261018/episode_000036/learned_feedback_v2/gain_.03"
"$task_root/.venv/bin/python" "$task_root/code/search_pinky_margins_v2.py" \
 --checkpoint "$task_actor/training/policy.npz" --base-module learned_feedback_policy --base-class LearnedPolicy \
 --scene "$task_actor/evaluation/execution/scene_physics.xml" \
 --episode-dir "$task_root/episodes/episode_000036" --baseline-report "$task_actor/evaluation/result.json" \
 --output "$task_root/refinement_20261018/episode_000036/learned_pinky_margins" \
 --hands both --right-startup-only --samples 128 --iterations 18 --workers 16 --seed 3661 --tolerance-deg 1.15
