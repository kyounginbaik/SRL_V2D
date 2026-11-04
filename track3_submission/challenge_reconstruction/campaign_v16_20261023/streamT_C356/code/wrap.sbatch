#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=T-wrap
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/logs/wrap-%j.log
# usage: sbatch wrap.sbatch EP DISTILLED_GRID_DIR OUT_GRID_DIR KEYS DONOR_POLICY  (streamA wrap_keys.py over every actor of a grid, 8 parallel, node01)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/code/env.sh
E=$1; G=$2; O=$3; K=$4; D=$5; cd /tmp
ls -d $G/actor_* | xargs -P 8 -I{} sh -c "$PY -B -u $C/streamA_spoon/code/wrap_keys.py $E {} $O/\$(basename {})_$K --keys $K --donor $D 2>&1 | tail -1"
