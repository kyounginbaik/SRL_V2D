#!/usr/bin/env bash
#SBATCH --job-name=dex-grasp-transfer2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-7%8
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/transfer-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episode=3
anchors=(4.0 4.3)
if ((SLURM_ARRAY_TASK_ID>=4)); then episode=8; anchors=(4.0 6.0); fi
anchor=${anchors[$(((SLURM_ARRAY_TASK_ID%4)/2))]}
blends=(0.3 0.6)
blend=${blends[$((SLURM_ARRAY_TASK_ID%2))]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/transfer_observed_grasp.py" --episode "$episode" --anchor "$anchor" --blend-s "$blend"
