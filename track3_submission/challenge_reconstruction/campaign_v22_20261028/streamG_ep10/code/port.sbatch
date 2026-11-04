#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=G10-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/logs/port-%j.log
# usage: sbatch port.sbatch LISTFILE   (each line: OUTNAME TEACHER_DIR [port_g.py extra args]); outputs streamG_ep10/port/OUTNAME
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/code/env.sh
EPD=$SD/world/episode_target/episode_000010; mkdir -p $SD/port
cd /tmp && cat $1 | xargs -P 16 -L 1 bash -c 'n=$0; t=$1; shift; '"$PY"' -B '"$SD"'/code/port_g.py 10 $t '"$EPD"' '"$SD"'/port/$n --pair '"$P10"' "$@" 2>&1 | grep -v Warn | tail -1 | sed "s/^/$n /"'
