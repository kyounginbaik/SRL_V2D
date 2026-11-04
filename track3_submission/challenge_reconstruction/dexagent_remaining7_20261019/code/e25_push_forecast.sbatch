#!/bin/bash
#SBATCH --job-name=e25-pushforecast
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/e25_pushforecast_%j.log
export E25_WORKSPACE=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3
export E25_HORIZON_GEOMETRY=1
export OPENBLAS_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/episode25_push_forecast.py --output "$E25_WORKSPACE/refinement/episode_000025/push_forecast_v4" --workers 8
