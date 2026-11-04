#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=board25-slots
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/board25-slots-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
/home/exaFLOPs/miniforge3/bin/python -u "$task_root/code/repair_board_slots.py" --object wooden_piece_1
/home/exaFLOPs/miniforge3/bin/python -u "$task_root/code/repair_board_slots.py" --object wooden_piece_2
/home/exaFLOPs/miniforge3/bin/python -u "$task_root/code/repair_board_slots.py" --merge
