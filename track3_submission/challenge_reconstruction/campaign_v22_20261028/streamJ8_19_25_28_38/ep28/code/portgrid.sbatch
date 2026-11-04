#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-28-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/logs/port-%j.log
# usage: sbatch portgrid.sbatch TEACHER OUTBASE "NAME|ARGS" ...   (runs port_d.py variants in parallel)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/code/env.sh
T=$1; OB=$2; shift 2; EPD=$SD/${WORLD:-world}/episode_target/episode_000028; mkdir -p $OB
cd /tmp
for v in "$@"; do N=${v%%|*}; A=${v#*|}; ( $PY -B -u $SD/code/port_d.py 28 $T $EPD $OB/$N $A 2>&1 | grep -v Warn | tail -2 ) & done; wait
