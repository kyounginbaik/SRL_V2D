#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=L10-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/logs/stage-%j.log
# usage: sbatch --export=ALL,DIST=<distilled actor dir>,BASE=<new staging base>,TEACHER=<teacher dir> stage.sbatch
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${DIST:?}" "${BASE:?}" "${TEACHER:?}"; EXTRA=""; [ -f $TEACHER/settings.json ] && EXTRA="--lineage-file $TEACHER/settings.json"
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10
cd /tmp && $R/.venv/bin/python -B -u $S/code/stage_L_component.py --episode 10 --distilled $DIST --base $BASE \
  --lineage-file $S/world_v1/recon_cup_thin_wall.json --lineage-file $S/code/build_recon_world.py --lineage-file $S/world_v1/source_assets/blue_cup.urdf --lineage-file $S/world_v1/source_assets/beige_cup.urdf \
  --lineage-file $S/retarget/nest4_teacher/result.json --lineage-file $TEACHER/result.json $EXTRA --lineage-file $S/code/retention.sbatch --lineage-file $S/code/actor_grid_L.py
