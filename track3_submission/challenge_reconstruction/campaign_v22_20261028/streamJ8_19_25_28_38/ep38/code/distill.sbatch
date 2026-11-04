#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8e38-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/logs/distill-%j.log
# usage: sbatch distill.sbatch EP TEACHER_DIR OUT_DIR [SEED] [GRID...]   (plain LearnedPolicy grid, streamC actor_grid_c.py, node01, stream D episode dir)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/env.sh
EPD=$SD/world0/episode_target/$(printf episode_%06d $1)
E=$1; TE=$2; O=$3; SEED=${4:-44}; shift 4 2>/dev/null || shift $#
GRID=${@:-512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002}
cd /tmp && $PY -B -u $C/streamC_rigid_ilc/tools/actor_grid_c.py --episode $E --teacher $TE --output $O --episode-dir $EPD --workers 8 --seed $SEED --grid $GRID
