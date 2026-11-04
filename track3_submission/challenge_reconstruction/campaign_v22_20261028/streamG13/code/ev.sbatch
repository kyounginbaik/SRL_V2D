#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=8G -t 00:45:00 -J G13-ev -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/logs/ev_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/${JOBS}); set -- $LINE
$G/code/py.sh ev.py $1 $2 $3
