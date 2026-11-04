#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=render25
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/render25-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/render_selected.py --episode 25 --baseline /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/refinement/episode_000025/push_forecast_v4 --refined /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/refinement/episode_000025/push_forecast_v4/actor_grid_best_effort_s45/actor_c1024_p0.001_v5e-05/evaluation --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/videos_v11/episode_000025 --stride 1 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025
