#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=T8-19save
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT8_ep19/logs/save-%j.log
# usage: sbatch save.sbatch ACTOR OUT TABLE SIDES  -> OUT/actor (baked phase weights + official evaluate_policy_component on node01)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
cd /tmp && $PY -B -u $J/code/actor_res19.py save --actor $1 --out $2 ${3:+--table $3} --sides ${4:-right} 2>&1 | grep --line-buffered -v Warn
