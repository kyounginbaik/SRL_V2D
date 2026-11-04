#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:45:00 -J G14-mk -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/logs/mk_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/${VLIST:-variants.txt})
MODE=${MODE:-frame} $G/code/py.sh make_public.py $LINE 2>&1 | grep -v Warn
