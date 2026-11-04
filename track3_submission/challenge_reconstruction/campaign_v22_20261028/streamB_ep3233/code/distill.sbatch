#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/distill-%j.log
# usage: sbatch distill.sbatch EP TEACHER OUT EPD [GRID] [SEED=44]   (streamC actor_grid_c.py, as streamR distill_r.sbatch)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh; cd /tmp
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode $1 --teacher $2 --output $3 --episode-dir $4 --workers 8 --seed ${6:-44} ${5:+--grid $5}
