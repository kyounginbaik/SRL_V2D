#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=F16-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/logs/F16-stage-%j.log
# usage: sbatch stage.sbatch BASE_DIR   (BASE holds input_manifest.json (ZIP manifest copy) + actor_report.json)
WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/world_T
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
F=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/code
BASE=$1; shift
REC=$WORLD/source/episode_000016/target_retarget.json
cd /tmp && $PY -u $F/stage_seven_component_retargeted.py --episode 16 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world_v2.py --lineage-file $F/build_world_f16.py --lineage-file $F/tlib16.py --lineage-file $REC "$@"
