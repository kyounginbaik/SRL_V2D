#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=N8-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/logs/stage10-%j.log
# ep10 staging with the unchanged stream G wrapper (stage_G_component.py); lineage adds the N8 up-corrected world record.
# usage: sbatch --export=ALL,DIST=<actor dir>,BASE=<new staging base>,TEACHER=<teacher dir> stage10.sbatch
set -euo pipefail
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/code/env.sh
SN=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30
: "${DIST:?}" "${BASE:?}" "${TEACHER:?}"; EXTRA=""; [ -f $TEACHER/settings.json ] && EXTRA="--lineage-file $TEACHER/settings.json"
cd /tmp && $PY -B -u $SD/code/stage_G_component.py --episode 10 --distilled $DIST --base $BASE \
  --lineage-file $L/world_v1/recon_cup_thin_wall.json --lineage-file $L/code/build_recon_world.py --lineage-file $L/world_v1/source_assets/blue_cup.urdf --lineage-file $L/world_v1/source_assets/beige_cup.urdf \
  --lineage-file $SN/ep10/world_up/target_reference.json --lineage-file $SN/code/build_world_g10up.py --lineage-file $SD/code/glib.py --lineage-file $SN/code/port_n8.py --lineage-file $SN/code/kcem_n8.py \
  --lineage-file $TEACHER/result.json $EXTRA
