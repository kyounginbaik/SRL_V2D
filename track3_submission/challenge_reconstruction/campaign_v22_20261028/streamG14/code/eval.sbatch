#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:50:00 -J G14-ev -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/logs/ev_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14
V=$((SLURM_ARRAY_TASK_ID/4)); Q=$((SLURM_ARRAY_TASK_ID%4))
LINE=$(sed -n "$((V+1))p" $G/code/eval_jobs.txt); set -- $LINE
EPS=("1 3 5 6 8" "10 14 15 16 19" "20 25 28 30 32" "33 35 36 37 38")
MODE=$1 MINFRAC=$2 $G/code/py.sh make_eval.py $3 $4 $5 $6 $7 $8 ${EPS[$Q]} 2>&1 | grep -v Warn
