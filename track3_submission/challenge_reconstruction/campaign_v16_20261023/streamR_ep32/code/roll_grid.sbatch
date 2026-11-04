#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=R32-roll
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/roll-%A_%a.log
# usage: sbatch --array=0-N roll_grid.sbatch GRIDFILE (lines: OUTNAME TEACHER CED DIR [--gain G])
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32; cd $R/code
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); name=${line%% *}; args=${line#* }
[ -f $S/roll/teachers/$name/result.json ] || $PY -B -u $S/code/roll_replan32.py ${args%% *} $S/roll/teachers/$name ${args#* }
