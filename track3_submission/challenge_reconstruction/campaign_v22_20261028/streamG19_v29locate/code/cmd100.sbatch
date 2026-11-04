#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 2 --mem=12G -t 00:50:00 -J G19-cmd -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/logs/cmd_%A_%a.log
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/code/${JOBS}); eval "$LINE"
