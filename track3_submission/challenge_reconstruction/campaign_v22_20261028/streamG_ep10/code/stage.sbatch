#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=G10-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/logs/stage-%j.log
# usage: sbatch --export=ALL,DIST=<actor dir>,BASE=<new staging base>,TEACHER=<teacher dir> stage.sbatch
set -euo pipefail
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG_ep10/code/env.sh
: "${DIST:?}" "${BASE:?}" "${TEACHER:?}"; EXTRA=""; [ -f $TEACHER/settings.json ] && EXTRA="--lineage-file $TEACHER/settings.json"
cd /tmp && $PY -B -u $SD/code/stage_G_component.py --episode 10 --distilled $DIST --base $BASE \
  --lineage-file $L/world_v1/recon_cup_thin_wall.json --lineage-file $L/code/build_recon_world.py --lineage-file $L/world_v1/source_assets/blue_cup.urdf --lineage-file $L/world_v1/source_assets/beige_cup.urdf \
  --lineage-file $SD/world/target_reference.json --lineage-file $SD/code/build_world_g.py --lineage-file $SD/code/glib.py --lineage-file $SD/code/port_g.py \
  --lineage-file $TEACHER/result.json $EXTRA
