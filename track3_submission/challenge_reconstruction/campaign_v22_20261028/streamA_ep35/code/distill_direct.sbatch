#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --job-name=A-ddist
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA_ep35/logs/ddist-%j.log
# usage: sbatch distill_direct.sbatch EP TEACHER OUT_GRID  (streamA2 distill_direct.py, fg x centers grid, node01)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA_ep35/code/env.sh
E=$1; TE=$2; O=$3; EPD=$ST/ep$E/world/episode_compose/$(printf episode_%06d $E); cd /tmp
printf "%s\n" "0.01 512" "0.03 512" "0.03 1024" "0.1 1024" "0.1 512" "0.3 1024" "0.03 2048" "0.1 2048" | xargs -P 8 -L 1 sh -c "$PY -B -u $C/streamA2_crotch/code/distill_direct.py --episode $E --teacher $TE --episode-dir $EPD --out $O/fg\$0_c\$1 --fg \$0 --centers \$1 2>&1 | tail -1"
