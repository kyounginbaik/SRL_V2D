#!/usr/bin/env bash
#SBATCH --job-name=dex-hold-grasp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-11%6
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/observed_hold/slurm-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(5 8)
anchors=(2.7 3.1 3.5)
modes=(fingers rigid)
episode=${episodes[$((SLURM_ARRAY_TASK_ID / 6))]}
anchor=${anchors[$(((SLURM_ARRAY_TASK_ID % 6) / 2))]}
mode=${modes[$((SLURM_ARRAY_TASK_ID % 2))]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/hold_observed_grasp.py" --episode "$episode" --anchor "$anchor" --mode "$mode"
