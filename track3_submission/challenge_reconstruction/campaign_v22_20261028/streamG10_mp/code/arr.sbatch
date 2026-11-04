#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=16G -t 00:50:00 -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/logs/%x_%A_%a.log
# usage: JOBS=file sbatch -J G10-x --array=0-N arr.sbatch   (each line = script + args)
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/$JOBS)
$G/code/py.sh $LINE
