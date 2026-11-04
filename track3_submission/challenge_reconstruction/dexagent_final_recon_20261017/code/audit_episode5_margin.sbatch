#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --job-name=ep5-margin-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-margin-audit-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$task_root/.software_gl/usr/lib64:${LD_LIBRARY_PATH:-}"
"$task_root/.venv/bin/python" "$task_root/code/audit_policy_component.py" --run "$task_root/refinement_20261018/pinky_margin_v2/episode_000005/evaluation" --baseline-report "$task_root/refinement_20261018/calibrated_scene_candidates/runs/episode_000005/original/result.json"
"$task_root/.venv/bin/python" "$task_root/code/audit_visual_self_contacts.py" --scene "$task_root/refinement_20261018/pinky_margin_v2/episode_000005/evaluation/execution/scene_physics.xml" --trajectory "$task_root/refinement_20261018/pinky_margin_v2/episode_000005/evaluation/execution/trajectory.npz" --output "$task_root/refinement_20261018/pinky_margin_v2/episode_000005/evaluation/audit/visual_self_contacts.json"
