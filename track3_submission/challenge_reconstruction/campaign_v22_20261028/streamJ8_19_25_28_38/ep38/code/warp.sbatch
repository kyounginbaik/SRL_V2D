#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8e38-warp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/logs/warp-%A_%a.log
# usage: sbatch --array=0-N warp.sbatch LISTFILE (each line: OUT_NAME TEACHER [warp args])
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/env.sh
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $L; N=$1; TE=$2; shift 2; mkdir -p $SD/warp
cd /tmp && $PY -B -u $SD/code/warp.py $TE $SD/warp/$N "$@" 2>&1 | grep -v Warn
