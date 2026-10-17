#!/usr/bin/env bash
#SBATCH --job-name=dex-cup-hold
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-8%9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/cup-hold-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
anchors=(1.9 2.0 2.2)
clearance=(0 .015 .025)
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/retain_cup_pickup.py" \
 --anchor "${anchors[$((SLURM_ARRAY_TASK_ID/3))]}" --clearance "${clearance[$((SLURM_ARRAY_TASK_ID%3))]}"
