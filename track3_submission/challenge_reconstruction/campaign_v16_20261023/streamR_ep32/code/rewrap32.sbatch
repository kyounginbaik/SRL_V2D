#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=R32-rewrap
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/rewrap-%A_%a.log
# usage: sbatch --array=0-N rewrap32.sbatch GRIDFILE   (one line per task: OUTNAME key=val ...)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32; cd /tmp
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); name=${line%% *}; args=${line#* }
[ -f $S/release/$name/evaluation/result.json ] || $PY -B -u $S/code/${RW:-rewrap32.py} $S/release/$name $args
