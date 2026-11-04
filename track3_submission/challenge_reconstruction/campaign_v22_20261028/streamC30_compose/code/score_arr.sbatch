#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 1 --mem=8G -t 00:40:00 -J C30-score -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/score_%A_%a.log
# usage: sbatch --array=1-N score_arr.sbatch LIST   (lines "EP DIR LABEL TARGET")
set -euo pipefail
read EP DIR LABEL TGT < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $1)
[ -f /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/scores/cand/$LABEL/score.json ] && { echo exists; exit 0; }
/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh $EP $DIR $LABEL $TGT
