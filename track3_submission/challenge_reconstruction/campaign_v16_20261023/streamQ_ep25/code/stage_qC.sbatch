#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=Q25-stageC
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25/logs/stage-%j.log
# Compose staging through the corrected-track wrapper (board_track_correction.json in the world's corrected source).
# usage: sbatch stage_q.sbatch WORLD BASE_DIR [--lineage-file ...]   (BASE_DIR holds actor_report.json + input_manifest.json)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25
W=$1; BASE=$2; shift 2
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u $Q/code/stage_seven_component_corrected_boards.py --episode 25 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $W/lineage.json --lineage-file $W/source_corrected/episode_000025/board_track_correction.json --lineage-file $Q/code/build_worldC.py --lineage-file $Q/code/compose_to_world.py --lineage-file $Q/code/make_slotted_mesh_C.py --lineage-file $Q/code/cut_slot_body.py --lineage-file $Q/meshes_slotted_C/slotted_mesh_report.json "$@"
