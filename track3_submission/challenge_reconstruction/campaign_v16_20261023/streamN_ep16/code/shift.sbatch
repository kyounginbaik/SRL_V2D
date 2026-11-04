#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=N16-shift
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/shift-%A_%a.log
# usage: sbatch --array=.. shift.sbatch OLD_WORLD NEW_WORLD OUT_BASE "TEACHER NAME TOUT" ...
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
OW=$1; NW=$2; O=$3; shift 3; A=("$@"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $N/code/shift_teacher.py $1 $OW $NW $O/$2 $3
