#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=T8-19ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT8_ep19/logs/ilc-%j.log
# usage: sbatch [--nodelist=..] ilc.sbatch SCRIPT args...
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
T8=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT8_ep19
S=$1; shift
cd /tmp && $PY -B -u $T8/code/$S "$@" 2>&1 | grep --line-buffered -v Warn
