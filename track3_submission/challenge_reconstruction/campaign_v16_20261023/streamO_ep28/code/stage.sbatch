#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=O28-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/logs/stage-%j.log
# Compose staging (package_code/stage_seven_component.py through the corrected-track wrapper ep30_v8/code/stage_seven_component_corrected_track.py,
# which only relaxes the ZIP-pose identity check using the correction record in the world's source dir). One stage job per BASE dir.
# usage: WORLD=... sbatch stage.sbatch BASE_DIR   (BASE_DIR holds actor_report.json + input_manifest.json; extra args = --lineage-file ...)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/code/env.sh
BASE=$1; shift
WRAP=$RS/ep30_v8/code/stage_seven_component_corrected_track.py
if [ -f $WORLD/collision_recon_corrected/episode_000028/sweeper_track_correction.json ]; then CORR="--lineage-file $WORLD/collision_recon_corrected/episode_000028/sweeper_track_correction.json"; else WRAP=$R7/package_code/stage_seven_component.py; CORR=""; fi
cd /tmp && $PY -u $WRAP --episode 28 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $O/code/build_world.py $CORR "$@"
