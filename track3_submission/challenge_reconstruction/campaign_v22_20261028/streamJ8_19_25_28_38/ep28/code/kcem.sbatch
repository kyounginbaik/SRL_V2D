#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-28-kcem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/logs/kcem-%j.log
# usage: sbatch kcem.sbatch BASE OUTNAME [kcem.py args]  -> ep28/cem/OUTNAME
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/code/env.sh
B=$1; N=$2; shift 2
cd /tmp && $PY -B -u $SD/code/kcem.py $B $SD/${WORLD:-world}/episode_target/episode_000028 $SD/cem/$N --workers 32 "$@" 2>&1 | grep -v Warn
