#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --job-name=ep3-pass-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000003/pass-audit-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$task_root/.software_gl/usr/lib64:${LD_LIBRARY_PATH:-}"
"$task_root/.venv/bin/python" "$task_root/code/audit_and_render_cem.py" --episode 3 --run "$task_root/refinement_20261018/episode_000003/retention_after45"
"$task_root/.venv/bin/python" "$task_root/code/grasp_stability.py" --scene "$task_root/refinement_20261018/episode_000003/retention_after45/execution/scene_physics.xml" --trajectory "$task_root/refinement_20261018/episode_000003/retention_after45/execution/trajectory.npz" --output "$task_root/refinement_20261018/episode_000003/retention_after45/audit/grasp_stability_substeps.json"
