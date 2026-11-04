#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-lilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/lilc-%A_%a.log
# usage: sbatch --array=0-N lilc.sbatch GRIDFILE (lines: TEACHER OUT EPD [opts])
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh; cd $R/code
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1)
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/code/left_ilc32.py $line
