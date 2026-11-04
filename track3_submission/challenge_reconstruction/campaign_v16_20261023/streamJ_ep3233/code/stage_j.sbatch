#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=J-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/stage-%j.log
# env: EP DISTILLED BASE TEACHER LINEAGE (space-separated extra lineage files)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${EP:?}" "${DISTILLED:?}" "${BASE:?}" "${TEACHER:?}"
set -euo pipefail; cd /tmp; L=""; for f in ${LINEAGE:-}; do L="$L --lineage-file $f"; done
$PY -B -u $R/refinement_20261021_ep32/code/stage_3233_component.py --episode $EP --distilled $DISTILLED --base $BASE --teacher $TEACHER $L
