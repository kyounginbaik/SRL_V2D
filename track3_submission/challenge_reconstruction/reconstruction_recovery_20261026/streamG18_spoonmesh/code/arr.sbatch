#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=8G -t 00:50:00 -o /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/logs/%x_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/$JOBS)
echo "$LINE"; eval "$LINE"
