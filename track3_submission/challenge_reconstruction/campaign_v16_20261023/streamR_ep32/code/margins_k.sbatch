#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-mk
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/mk-%j.log
# env: ACTOR OUT [HAND=right TOL=5 STD=0.04 SEED=3333]  (streamK_ep35/code/search_margins_k.py, resumable)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${ACTOR:?}" "${OUT:?}"; cd /tmp; R0=""; [ -f $OUT/settings.json ] && R0=--resume
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/code/search_margins_k.py --episode 32 --actor $ACTOR --output $OUT --hand ${HAND:-right} --tolerance-deg ${TOL:-5} --std ${STD:-0.04} --samples 96 --iterations 12 --workers 16 --seed ${SEED:-3333} $R0
