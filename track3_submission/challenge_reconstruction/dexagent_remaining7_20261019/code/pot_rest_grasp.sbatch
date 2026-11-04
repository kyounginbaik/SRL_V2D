#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=pot-rest-grasp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/pot-rest-grasp-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest05 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest05_early --close-shift=-0.15 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest02 --rest-time=.2 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest05_low --local-offset 0 0 -0.007 &
wait
