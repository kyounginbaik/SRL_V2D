#!/usr/bin/env bash
#SBATCH --job-name=dex-cup-feedback
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/cup-feedback-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
anchors=(2.5 3.5)
clearance=(0 .015)
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/cup_feedback_transfer.py" \
 --anchor "${anchors[$((SLURM_ARRAY_TASK_ID/2))]}" --clearance-m "${clearance[$((SLURM_ARRAY_TASK_ID%2))]}" --deadline "${CUP_DEADLINE:-1.95}"
