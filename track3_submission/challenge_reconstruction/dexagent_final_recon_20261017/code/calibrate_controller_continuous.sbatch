#!/usr/bin/env bash
#SBATCH --job-name=dex-pd-angle
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-24%8
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/controller_sweep_continuous/slurm-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episodes=(1 3 5 8 10)
variants=(baseline kp5000_r200_d5 kp5000_r200_d10 kp10000_r200_d5 kp10000_r200_d10)
episode=${episodes[$((SLURM_ARRAY_TASK_ID / 5))]}
variant=${variants[$((SLURM_ARRAY_TASK_ID % 5))]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/calibrate_controller.py" --episode "$episode" --variant "$variant" --continuous-angles --output-root "$task_root/refinement_20261017/controller_sweep_continuous"
