#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-19kcem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/logs/kcem-%j.log
# usage: sbatch kcem.sbatch BASE OUTNAME [kcem19.py args] -> ep19/cem/OUTNAME
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
B=$1; N=$2; shift 2
EPD=$(python3 -c "import json;print(json.load(open('$B/result.json'))['input_dir'])")   # world of the base teacher
echo EPD $EPD
cd /tmp && $PY -B -u $J/code/kcem19.py $B $EPD $J/cem/$N --workers 16 "$@" 2>&1 | grep --line-buffered -v Warn
