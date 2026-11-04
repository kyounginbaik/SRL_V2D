#!/usr/bin/env bash
#SBATCH --job-name=dex-grasp-polish
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/polish-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(1 3 5 8 10)
episode=${episodes[$SLURM_ARRAY_TASK_ID]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/polish_grasp_blocks.py" --episode "$episode" --workers "$SLURM_CPUS_PER_TASK" --samples "${POLISH_SAMPLES:-64}" --iterations "${POLISH_ITERATIONS:-8}" --output-root "${POLISH_OUTPUT_ROOT:-$task_root/refinement_20261017/block_polish}"
