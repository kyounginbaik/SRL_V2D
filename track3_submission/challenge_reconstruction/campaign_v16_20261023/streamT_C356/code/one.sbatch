#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=T-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/logs/one-%j.log
# usage: sbatch one.sbatch "COMMAND ..."   (command run with env.sh sourced, cwd /tmp)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/code/env.sh
cd /tmp && eval "$1"
