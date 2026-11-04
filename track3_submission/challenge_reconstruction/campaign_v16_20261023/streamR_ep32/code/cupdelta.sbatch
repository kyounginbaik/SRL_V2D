#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=R32-cupd
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/cupd-%A_%a.log
# usage: sbatch --array=0-N cupdelta.sbatch GRIDFILE (lines: OUTDIR args...)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh; cd $R/code
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); o=${line%% *}; args=${line#* }
[ -f $o/result.json ] || $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/code/cup_delta_seed.py ${args%% *} $o ${args#* }
