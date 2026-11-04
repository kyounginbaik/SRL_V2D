#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=T-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/logs/distill-%j.log
# usage: sbatch distill.sbatch EP TEACHER_DIR OUT_DIR [SEED]   (plain LearnedPolicy grid, streamC actor_grid_c.py, node01, compose episode dir)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/code/env.sh
EPD=$ST/ep$1/world/episode_compose/$(printf episode_%06d $1)
cd /tmp && $PY -B -u $C/streamC_rigid_ilc/tools/actor_grid_c.py --episode $1 --teacher $2 --output $3 --episode-dir $EPD --workers 8 --seed ${4:-44} --grid 512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002 2048,0.01,0.0002 1024,0.03,0.0005
