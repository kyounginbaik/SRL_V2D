#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=T8-19cem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT8_ep19/logs/cem-%j.log
# usage: sbatch cem.sbatch actor_res19.py cem args...  (search only, --no-save; actors are saved on node01 with save.sbatch)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
cd /tmp && $PY -B -u $J/code/actor_res19.py "$@" --workers 16 --no-save 2>&1 | grep --line-buffered -v Warn
