#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=N8-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE TEACHER_DIR [extra args]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
E=$1; A=$2; B=$3; TE=$4; shift 4
cd /tmp && $PY -B -u $SN/code/stage_n8.py --episode $E --distilled $A --base $B --teacher $TE "$@"
