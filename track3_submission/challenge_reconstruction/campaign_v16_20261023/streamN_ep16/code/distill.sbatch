#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/distill-%j.log
# plain learned_feedback_policy.LearnedPolicy grid (Stream C actor_grid_c.py), pinned to node01.
# usage: WORLD=... sbatch distill.sbatch TEACHER_DIR OUT_DIR [--seed S] [--grid c,p,v ...]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
T=$1; O=$2; shift 2
cd /tmp && $PY -B -u $C/streamC_rigid_ilc/tools/actor_grid_c.py --episode 16 --teacher $T --output $O --episode-dir $EPDIR --workers 8 --grid 512,0.001,0.00005 1024,0.001,0.00005 2048,0.001,0.00005 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0002 1024,0.01,0.0002 2048,0.01,0.0002 1024,0.03,0.0005 "$@"
