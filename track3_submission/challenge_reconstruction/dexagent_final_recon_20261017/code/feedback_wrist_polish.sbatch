#!/usr/bin/env bash
#SBATCH --job-name=dex-wrist-feedback
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-4%5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/feedback-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(1 3 5 8 10)
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/feedback_wrist_polish.py" --episode "${episodes[$SLURM_ARRAY_TASK_ID]}" --workers "$SLURM_CPUS_PER_TASK"
