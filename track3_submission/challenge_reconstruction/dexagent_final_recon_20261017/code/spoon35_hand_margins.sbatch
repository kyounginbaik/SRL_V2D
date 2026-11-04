#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --job-name=spoon35-finger-fit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon35-fingers-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
run_root="$task_root/refinement_20261018"
"$task_root/.venv/bin/python" "$task_root/code/search_hand_margins.py" \
 --checkpoint "$run_root/learned_spoon35_chart/gain_0.03/training/policy.npz" \
 --scene "$run_root/learned_spoon35_chart/gain_0.03/evaluation/execution/scene_physics.xml" \
 --episode-dir "$run_root/wrist_chart/episode_000035/chart_0/adapter" \
 --baseline-report "$run_root/learned_spoon35_chart/gain_0.03/evaluation/result.json" \
 --config "$run_root/spoon35_hand_margins/config.json" --output "$run_root/spoon35_hand_margins/search" \
 --samples 64 --iterations 12 --workers 16 --seed 3503 --tolerance-deg 1.1
