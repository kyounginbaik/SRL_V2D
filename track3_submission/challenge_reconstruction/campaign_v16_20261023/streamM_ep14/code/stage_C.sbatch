#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=M14-stageC
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/logs/stage-%j.log
# Compose staging for the compose-retargeted world (unchanged stage_seven_component.py through the corrected-target wrapper, which only
# replaces the ZIP-pose identity check by the compose_target_correction.json record). One stage job per BASE dir.
# usage: WORLD=world_C sbatch stage_C.sbatch BASE_DIR
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/code/env.sh
BASE=$1; shift; C=$M/compose/source_C/episode_000014/compose_target_correction.json
cd /tmp && $PY -u $M/code/stage_seven_component_compose_target.py --episode 14 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $M/code/build_world.py --lineage-file $M/code/build_world_C.py --lineage-file $M/code/compose_reference.py --lineage-file $C "$@"
