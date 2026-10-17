#!/usr/bin/env bash
#SBATCH --job-name=dex-grasp-cem
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/slurm-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(1 3 5 8 10)
episode=${episodes[$SLURM_ARRAY_TASK_ID]}
printf -v episode_name 'episode_%06d' "$episode"
run_name=${CEM_RUN_NAME:-cem_v1}
samples=${CEM_SAMPLES:-96}
iterations=${CEM_ITERATIONS:-10}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/refine_grasp_cem.py" \
 --episode "$episode" --scene "$task_root/refinement_20261017/controller_sweep_continuous/$episode_name/kp5000_r200_d5/controller_scene.xml" \
 --reference "$task_root/refinement_20261017/geometry_v2/$episode_name/tracking_reference_geometry.npz" \
 --output "$task_root/refinement_20261017/$run_name/$episode_name" \
 --samples "$samples" --iterations "$iterations" --workers "$SLURM_CPUS_PER_TASK" --seed 44
