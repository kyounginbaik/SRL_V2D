#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=ep1-contact-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep1-contact-audit-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
selected="$task_root/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_carry_refinement_v2/evaluation"
output="$task_root/refinement_20261018/shared_policy_handoff/episode_000001/validation/physical_quality/grasp_stability_substeps.json"
"$task_root/.venv/bin/python" "$task_root/code/grasp_stability.py" --scene "$selected/execution/scene_physics.xml" --trajectory "$selected/execution/trajectory.npz" --output "$output"
