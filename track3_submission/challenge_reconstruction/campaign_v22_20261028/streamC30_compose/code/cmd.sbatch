#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 2 --mem=16G -t 00:48:00 -J C30-cmd -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/cmd_%j.log
# usage: sbatch cmd.sbatch "<shell command>"
cd /tmp; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}; eval "$1"
