#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-6%7
#SBATCH --job-name=dexagent-hawor-seven
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/mpc-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
task_python=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
episodes=(14 15 16 20 25 28 30)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task_root"
"$task_python" -u code/prepare_seven.py --episodes "${episodes[$SLURM_ARRAY_TASK_ID]}"
"$task_python" -u code/run_seven_mpc.py --episode "${episodes[$SLURM_ARRAY_TASK_ID]}" --tag hawor_mpc_v1 --workers 4 --samples 16 --max-minutes 10
