#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/margins-%j.log
# env: ACTOR OUT WRIST [STD=0.03 BOUND=0.15 PSTD=0.003 PBOUND=0.015 SEED=3333 TOL=5]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${ACTOR:?}" "${OUT:?}" "${WRIST:?}"; cd /tmp
R0=""; [ -f $OUT/settings.json ] && R0=--resume
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/code/search_margins_r.py --episode 32 --actor $ACTOR --output $OUT --wrist $WRIST --std ${STD:-0.03} --bound ${BOUND:-0.15} --pos-std ${PSTD:-0.003} --pos-bound ${PBOUND:-0.015} --tolerance-deg ${TOL:-5} --samples 96 --iterations 12 --workers 16 --seed ${SEED:-3333} $R0
