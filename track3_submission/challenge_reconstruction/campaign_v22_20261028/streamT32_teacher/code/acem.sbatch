#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-acem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/acem-%A_%a.log
# usage: sbatch --array=0-N acem.sbatch LIST (line: acem.py args). Run on AMD nodes (node100/node01: same CPU as the evaluator node).
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); export ACEM_TMP=/tmp
cd /tmp; $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT32_teacher/code/acem.py $line
