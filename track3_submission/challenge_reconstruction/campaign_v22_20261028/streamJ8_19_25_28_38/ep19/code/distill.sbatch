#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-19distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/logs/distill-%j.log
# usage: sbatch distill.sbatch TEACHER OUT WORLD_DIR SEED GRID...   (streamC actor_grid_c.py, unchanged)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
T=$1; O=$2; W=$3; S=$4; shift 4
cd /tmp && $PY -B -u $C/streamC_rigid_ilc/tools/actor_grid_c.py --episode 19 --teacher $T --output $O --episode-dir $W/episode_target/episode_000019 --workers 4 --seed $S --grid "$@" 2>&1 | grep --line-buffered -v Warn
