#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=N8-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/logs/grid-%A_%a.log
# usage: sbatch --array=0-K grid.sbatch CMDFILE   (line SLURM_ARRAY_TASK_ID+1 is evaluated; env.sh sourced, cwd /tmp)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); echo "$L"; cd /tmp && eval "$L"
