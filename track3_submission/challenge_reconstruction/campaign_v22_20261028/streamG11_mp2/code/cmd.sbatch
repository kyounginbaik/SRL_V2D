#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=16G -t 00:50:00 -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG11_mp2/logs/%x_%A_%a.log
# usage: JOBS=file sbatch -J G10-x --array=0-N cmd.sbatch   (each line = a full shell command)
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG11_mp2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/$JOBS)
eval "$LINE"
