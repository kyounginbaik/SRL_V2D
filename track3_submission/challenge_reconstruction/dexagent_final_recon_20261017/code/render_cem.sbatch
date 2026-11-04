#!/usr/bin/env bash
#SBATCH --job-name=cem-audit-render
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:45:00
#SBATCH --array=0-4%3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem-render-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LP_NUM_THREADS=2
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(1 3 5 8 10)
episode=${episodes[$SLURM_ARRAY_TASK_ID]}
experiment_tag=${CEM_RENDER_TAG:-cem_v1}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/audit_and_render_cem.py" \
  --episode "$episode" --tag "$experiment_tag"
