#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=D-one
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD_ep1_20/logs/one-%j.log
# usage: sbatch one.sbatch "COMMAND ..."   (env.sh sourced, cwd /tmp)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD_ep1_20/code/env.sh
cd /tmp && eval "$1"
