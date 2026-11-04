#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=T-ret
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/logs/ret-%j.log
# usage: sbatch ret.sbatch EP SRC OUT START STOP SIDE PROTECT [SEED] [ITERS]  (refine_versioned_retention_full.py, compose episode dir)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamT_C356/code/env.sh
EPD=$ST/ep$1/world/episode_compose/$(printf episode_%06d $1)
cd $R/code && $PY -B -u refine_versioned_retention_full.py --source $2 --episode-dir $EPD --output $3 --start-s $4 --stop-s $5 --side $6 --protect-object $7 --samples 96 --iterations ${9:-14} --workers 24 --seed ${8:-63}
