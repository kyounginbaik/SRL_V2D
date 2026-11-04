#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --array=0-7%3
#SBATCH --job-name=expanded-native-mpc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/mpc-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd "$task_root"
episodes=(6 19 32 33 35 36 37 38)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python code/expanded_native_mpc.py --episode "${episodes[$SLURM_ARRAY_TASK_ID]}"
