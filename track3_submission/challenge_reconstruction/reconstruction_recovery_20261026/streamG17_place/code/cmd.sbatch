#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:48:00 -J G17-cmd -o /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG17_place/logs/cmd_%A_%a.log
cd /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG17_place/code
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${JOBS}); eval "$LINE"
