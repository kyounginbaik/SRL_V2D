#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=v22-score
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/logs/score-%A_%a.log
# usage: sbatch --array=1-N score_array.sbatch LIST   (lines "EP DIR LABEL")
set -euo pipefail
read EP DIR LABEL < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $1)
/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh $EP $DIR $LABEL
