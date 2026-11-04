#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-19res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/logs/res-%j.log
# usage: sbatch res.sbatch actor_res19.py args...   (AMD node: final actor evaluation must match build v2)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
cd /tmp && $PY -B -u $J/code/actor_res19.py "$@" --workers 16 2>&1 | grep --line-buffered -v Warn
