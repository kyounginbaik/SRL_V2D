#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sweeper30-rack
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-rack-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
"/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python" -u "$task_root/code/refine_rack_hold.py" \
 --source-run "$task_root/physics_collision_v2/sweeper_transfer/donor3_park0_v1/pickup_search_v1" \
 --output "$task_root/physics_collision_v2/sweeper_transfer/rack_hold_v1" \
 --samples 64 --iterations 12 --workers 8 --seed 45
