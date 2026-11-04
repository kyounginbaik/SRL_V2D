#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-stage2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/stage-%j.log
# Compose staging for the re-targeted-spoon world: stage_seven_component.py (unchanged) through code/stage_seven_component_retargeted.py,
# which only relaxes the ZIP-pose identity check using the world's spoon_track_retarget.json record. One stage job per BASE dir.
# usage: WORLD=... sbatch stage_retarget.sbatch BASE_DIR
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
BASE=$1; shift
REC=$WORLD/collision_recon/episode_000016/spoon_track_retarget.json
cd /tmp && $PY -u $N/code/stage_seven_component_retargeted.py --episode 16 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world_v2.py --lineage-file $N/code/retarget_spoon_track.py --lineage-file $REC "$@"
