#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=M8-stage15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/logs/stage-%j.log
# usage: sbatch stage15.sbatch BASE_DIR   (streamS newtrack stager unchanged, world_6prong_nt)
WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/world_6prong_nt
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/code/env.sh
BASE=$1; shift
cd /tmp && $PY -u $N/code/stage_seven_component_newtrack.py --episode 15 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world.py --lineage-file $N/code/port_specs.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code/mres.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code/mlib.py "$@"
