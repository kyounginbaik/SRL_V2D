#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=M14-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/logs/stage-%j.log
# Compose staging with the unchanged package_code/stage_seven_component.py (episode 14 has no track correction). One stage job per BASE dir.
# usage: WORLD=... sbatch stage.sbatch BASE_DIR   (BASE_DIR holds actor_report.json + input_manifest.json; extra args = --lineage-file ...)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/code/env.sh
BASE=$1; shift
cd /tmp && $PY -u $R7/package_code/stage_seven_component.py --episode 14 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $M/code/build_world.py "$@"
