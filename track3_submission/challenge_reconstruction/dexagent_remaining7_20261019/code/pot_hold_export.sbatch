#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=pot-hold-export
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/pot-hold-export-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
"$py" -u "$base/code/pot_hold_export.py" --source "$base/diagnostics/pot_transfer/pinch_roll180" --output "$base/diagnostics/pot_transfer/pinch_roll180_hold" &
"$py" -u "$base/code/pot_hold_export.py" --source "$base/diagnostics/pot_transfer/pinch_roll270" --output "$base/diagnostics/pot_transfer/pinch_roll270_hold" &
wait
