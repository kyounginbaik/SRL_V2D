#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sweeper30-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-pickup-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
task_seed="$task_root/physics_collision_v2/sweeper_transfer/donor3_park${SWEEPER_PARK:-0}_v1"
"$legacy_root/.venv/bin/python" -u "$legacy_root/code/physics_pickup_search_v2.py" \
  --spec "$task_seed/spec.json" --output "$task_seed/pickup_search_v1" \
  --workers 8 --samples 48 --iterations 12 --seed 44
