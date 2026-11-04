#!/usr/bin/env bash
#SBATCH --job-name=spoon-neck-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/spoon-neck-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/run_spoon_collision_feedback.py" --rank "$SLURM_ARRAY_TASK_ID"
