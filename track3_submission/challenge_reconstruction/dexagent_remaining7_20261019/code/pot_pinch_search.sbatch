#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=pot-pinch-search
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/pot-pinch-search-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
"$py" -u "$base/code/pot_pinch_search.py" --spec "$base/diagnostics/pot_transfer/rest_roll$POT_ROLL/spec.json" --output "$base/diagnostics/pot_transfer/pinch_roll$POT_ROLL" --seed "$POT_SEED" --workers 16 --samples 96 --iterations 12
