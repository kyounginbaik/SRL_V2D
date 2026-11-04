#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=N8-kcem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/logs/kcem-%j.log
# usage: sbatch kcem.sbatch BASE_TEACHER EPD OUT [kcem_n8.py args]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
B=$1; E=$2; O=$3; shift 3
cd /tmp && $PY -B -u $SN/code/kcem_n8.py $B $E $O --workers 24 "$@" 2>&1 | grep -v Warn
