#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=F6-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/logs/F6-port-%j.log
# usage: sbatch port.sbatch EP TEACHER OLD_EPD OUTROOT VARFILE   (VARFILE lines: NAME port.py-args...; 16 in parallel)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep6/code/env.sh
E=$1; TE=$2; OLD=$3; OR=$4; VF=$5; cd /tmp
grep -v '^#' $VF | grep . | xargs -P 16 -L 1 sh -c "$PY -B -u $ST/ep6/code/port.py $E $TE $OLD $OR/\$0 \"\$@\" 2>&1 | tail -1" 
