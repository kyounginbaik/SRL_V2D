#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon15-retention
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon15-retention-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/refine_expanded_retention.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/collision_v2_donor28_park1 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_after30 --start-s 3.0 --stop-s 5.2 --protect-object plastic_dish_rack --seed 52 --scale 0.75 --samples 64 --iterations 8 --workers 16
