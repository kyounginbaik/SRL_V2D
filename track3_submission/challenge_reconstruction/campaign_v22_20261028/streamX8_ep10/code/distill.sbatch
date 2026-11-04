#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=X8-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamX8_ep10/logs/distill-%j.log
# usage: sbatch distill.sbatch EP EPISODE_DIR TEACHER_DIR OUT_DIR [SEED] [GRID...]   (plain LearnedPolicy grid, streamC actor_grid_c.py, node01)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
E=$1; EPD=$2; TE=$3; O=$4; SEED=${5:-44}; shift 5 2>/dev/null || shift $#
GRID=${@:-512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002}
cd /tmp && $PY -B -u $C/streamC_rigid_ilc/tools/actor_grid_c.py --episode $E --teacher $TE --output $O --episode-dir $EPD --workers 8 --seed $SEED --grid $GRID
