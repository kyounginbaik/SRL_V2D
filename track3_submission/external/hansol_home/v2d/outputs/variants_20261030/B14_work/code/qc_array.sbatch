#!/bin/bash
#SBATCH --job-name=B14-qc
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hansol/v2d/outputs/variants_20261030/B14_work/logs/qc-%A_%a.log
# Variant B: dense GT-free image-evidence QC (team tools/reconstruction_audit/qc.py, unchanged); LIST lines "<episode root> <poses folder>"
set -euo pipefail
cd /data1/SRL_V2D
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${LIST:?}")
/home/hansol/v2d/.venvs/v2d-host/bin/python -B tools/reconstruction_audit/qc.py $LINE
