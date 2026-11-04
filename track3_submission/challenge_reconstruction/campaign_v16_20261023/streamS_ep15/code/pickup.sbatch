#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=S15-pick
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/logs/pickup-%A_%a.log
# Pickup CEM on the full 8.25 s horizon (streamE rack_safe_pickup.py = physics_pickup_search_v2 + rack/rotation penalty).
# usage: sbatch --array=0-N pickup.sbatch JOBS_TXT    lines: OUT_DIR SPEC SEED SAMPLES ITER [RESUME_PROGRESS_JSON|-] [RACK_TOL] [ROT_WEIGHT]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/code/env.sh
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $line
OUT=$1; SPEC=$2; SEED=$3; SAM=$4; IT=$5; RES=${6:--}; TOL=${7:-0.8}; RW=${8:-1.5}
X=""; [ "$RES" != "-" ] && X="--resume $RES"
cd /tmp && $PY -B -u $C/streamE_ep15/code/rack_safe_pickup.py --spec $SPEC --output $OUT --seed $SEED --samples $SAM --iterations $IT --workers 16 --rack-tol-cm $TOL --rot-weight $RW $X
