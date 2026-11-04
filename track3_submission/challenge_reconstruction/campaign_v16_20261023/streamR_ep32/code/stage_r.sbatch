#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=R32-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/stage-%j.log
# env: DISTILLED BASE [EXTRA_LINEAGE]  (teacher = Stream J chainB/c2 = the v19 teacher; original reference, no track correction)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${DISTILLED:?}" "${BASE:?}"; set -euo pipefail; cd /tmp
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32
L=""; for f in $J/ep32/p1/T_relR10.9/result.json $J/ep32/chainB/c1_R_6.6_9.0/result.json $J/ep32/chainB/c2_R_9.0_11.0/result.json $J/ep32/margins/c2_c4096_cold/training/training_report.json \
  $S/code/rewrap32_v2.py $S/code/search_margins_r.py $S/diag/evidence_summary.json ${EXTRA_LINEAGE:-}; do L="$L --lineage-file $f"; done
$PY -B -u $R/refinement_20261021_ep32/code/stage_3233_component.py --episode 32 --distilled $DISTILLED --base $BASE --teacher $J/ep32/chainB/c2_R_9.0_11.0 $L
