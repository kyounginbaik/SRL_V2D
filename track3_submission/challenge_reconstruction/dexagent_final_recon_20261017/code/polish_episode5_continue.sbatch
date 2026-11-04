#!/usr/bin/env bash
#SBATCH --job-name=dex-spoon-polish
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=46-48%3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/continue-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/polish_grasp_blocks.py" \
 --episode 5 --workers "$SLURM_CPUS_PER_TASK" --samples 96 --iterations 10 --seed "$SLURM_ARRAY_TASK_ID" \
 --source-root "$task_root/refinement_20261017/block_polish" \
 --output-root "$task_root/refinement_20261017/block_continue_s$SLURM_ARRAY_TASK_ID"
