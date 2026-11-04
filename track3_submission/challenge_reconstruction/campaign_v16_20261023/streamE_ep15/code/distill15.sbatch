#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=E-distill15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/distill-%j.log
# usage: sbatch distill15.sbatch TEACHER_DIR OUT_DIR   (plain learned_feedback_policy.LearnedPolicy actors, 8-point grid)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 15 --teacher $1 --output $2 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015 --workers 8 --grid 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 1024,0.01,0.0005 2048,0.01,0.0005 512,0.01,0.0005 1024,0.03,0.0005 1024,0.001,0.00005
