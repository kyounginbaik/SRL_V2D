#!/usr/bin/env bash
#SBATCH --job-name=dex-grasp-v3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-5%6
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/seed_physics_v3/slurm-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(3 5 8)
variants=(baseline kp5000_r200_d5)
episode=${episodes[$((SLURM_ARRAY_TASK_ID / 2))]}
variant=${variants[$((SLURM_ARRAY_TASK_ID % 2))]}
printf -v episode_name 'episode_%06d' "$episode"
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/evaluate_grasp_seed.py" --episode "$episode" --variant "$variant" --reference "$task_root/refinement_20261017/geometry_v3/$episode_name/tracking_reference_geometry.npz" --output-root "$task_root/refinement_20261017/seed_physics_v3"
