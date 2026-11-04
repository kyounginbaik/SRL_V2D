#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=F6-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/logs/F6-one-%j.log
# usage: sbatch one.sbatch "COMMAND ..."   (command run with env.sh sourced, cwd /tmp)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep6/code/env.sh
cd /tmp && eval "$1"
