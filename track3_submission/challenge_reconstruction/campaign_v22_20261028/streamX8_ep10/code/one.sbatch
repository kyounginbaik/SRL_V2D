#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --job-name=X8-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamX8_ep10/logs/one-%j.log
# usage: sbatch one.sbatch SCRIPT ARGS...   (X8 single python job)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
X=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamX8_ep10
S=$1; shift; cd /tmp && $PY -B -u $X/code/$S "$@" 2>&1 | grep -v Warn
