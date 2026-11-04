#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=sB-ep3-pick
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep3-pick-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
names=(near_a near_b mid_a mid_b); name=${names[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/physics_pickup_search.py --spec /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/power_grasp/specs/episode_000003/$name.json --output /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/power_grasp/search/episode_000003/$name --samples 128 --iterations 20 --workers 16 --seed 44
