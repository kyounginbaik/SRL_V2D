#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=pot-roll-grasp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/pot-roll-grasp-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest_roll90 --roll-deg 90 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest_roll180 --roll-deg 180 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest_roll270 --roll-deg 270 &
"$py" -u "$base/code/pot_rest_grasp.py" --tag rest_roll135 --roll-deg 135 &
wait
