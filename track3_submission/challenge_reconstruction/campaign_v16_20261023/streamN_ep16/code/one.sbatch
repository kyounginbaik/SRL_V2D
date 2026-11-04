#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=N16-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/one-%j.log
# run one Stream-N tool command in the world environment. usage: WORLD=... sbatch one.sbatch TOOL.py ARGS...
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
cd /tmp && $PY -u "$@"
