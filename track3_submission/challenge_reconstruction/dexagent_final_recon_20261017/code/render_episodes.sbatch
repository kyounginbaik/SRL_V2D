#!/usr/bin/env bash
#SBATCH --job-name=dex-render
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/runs/pilot_v2/render-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LP_NUM_THREADS=2
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(000003 000005 000008 000010)
episode=${episodes[$SLURM_ARRAY_TASK_ID]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/render_rollouts.py" --args "$task_root/runs/pilot_v2/episode_${episode}/seed_44/videos/render_args.json"
