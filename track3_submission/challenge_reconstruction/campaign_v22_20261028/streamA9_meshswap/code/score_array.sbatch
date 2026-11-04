#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --job-name=A9-score
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/logs/score-%A_%a.log
# A9: score_any vs target_final_v6. usage: sbatch --array=1-N score_array.sbatch LIST (lines "EP DIR LABEL")
set -euo pipefail
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
read EP DIR LABEL < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $1)
$C/code/score_any.sh $EP $DIR $LABEL $C/target_final_v6
