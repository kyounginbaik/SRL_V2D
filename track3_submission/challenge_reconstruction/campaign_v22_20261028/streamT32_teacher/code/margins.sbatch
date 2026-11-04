#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-mk
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/mk-%j.log
# usage: sbatch margins.sbatch EP ACTOR OUT [HAND=right] [STD=0.04] [SEED=3333]  (streamK_ep35/code/search_margins_k.py unchanged, resumable)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
cd /tmp; R0=""; [ -f $3/settings.json ] && R0=--resume
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/code/search_margins_k.py --episode $1 --actor $2 --output $3 --hand ${4:-right} --tolerance-deg 5 --std ${5:-0.04} --samples 96 --iterations 12 --workers 16 --seed ${6:-3333} $R0
