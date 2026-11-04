#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8e38-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/logs/port-%A_%a.log
# usage: sbatch --array=0-N port.sbatch LISTFILE   (each line: OUT_NAME TEACHER_DIR [port_d args...])
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/env.sh
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $L; N=$1; TE=$2; shift 2
W=$SD/world0/episode_target/episode_000038
cd /tmp && $PY -B -u $SD/code/port_d.py 38 $TE $W $SD/port0/$N --pair right=mini_dust_pan "$@" 2>&1 | grep -v Warn
