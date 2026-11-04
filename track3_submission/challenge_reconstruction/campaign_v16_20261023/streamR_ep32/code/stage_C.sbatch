#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-stageC
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/stageC-%j.log
# env: DISTILLED BASE TEACHER [EXTRA_LINEAGE]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${DISTILLED:?}" "${BASE:?}" "${TEACHER:?}"; set -euo pipefail; cd /tmp
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32
L=""; for f in $S/code/compose_retarget32_v3.py $S/code/prepare_pickup32c.py $S/code/cup_delta_seed.py $S/code/rewrap32_v2.py $S/code/measure_hold_rotation.py ${EXTRA_LINEAGE:-}; do L="$L --lineage-file $f"; done
$PY -B -u $S/code/stage_ep32_compose_component.py --episode 32 --distilled $DISTILLED --base $BASE --teacher $TEACHER $L
