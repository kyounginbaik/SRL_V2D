#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sweeper30-cem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-cem-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
task_python=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D
"$task_python" -u "$task_root/code/refine_and_train.py" \
  --episode 30 --base "$task_root/physics_collision_v2" \
  --source-run "$task_root/physics_collision_v2/runs/baseline_v1/episode_000030/baseline" \
  --output "$task_root/physics_collision_v2/refinement/episode_000030/cem_v1" \
  --workers 8 --samples 32 --iterations 6 --local-iterations 6 --seed 44
