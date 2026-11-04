#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-28-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/logs/one-%j.log
# usage: sbatch one.sbatch "COMMAND ..."   (env.sh sourced, cwd /tmp)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28/code/env.sh
cd /tmp && eval "$1"
