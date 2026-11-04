#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=J8-19port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/logs/port-%A_%a.log
# usage: sbatch --array=0-N port.sbatch JOBFILE   (each line: TEACHER OUT ARGS... for port19J.py)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code/env.sh
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1)
cd /tmp && $PY -B -u $J/code/port19J.py $L 2>&1 | grep --line-buffered -v Warn
