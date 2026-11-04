#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-25stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep25/logs/stage-%j.log
# usage: sbatch stage.sbatch STAGING_BASE   (streamE stage_qC wrapper call, unchanged staging script, + J8 lineage)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25; E=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30
W=$E/ep25/world_T1; BASE=$1; J=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep25
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u $Q/code/stage_seven_component_corrected_boards.py --episode 25 --base $BASE --actor-report $BASE/actor_report.json --lineage-file $W/lineage.json --lineage-file $W/source_corrected/episode_000025/board_track_correction.json --lineage-file $E/code25/build_worldE.py --lineage-file $E/code/conv25.py --lineage-file $E/code/elib.py --lineage-file $Q/code/make_slotted_mesh_C.py --lineage-file $Q/code/cut_slot_body.py --lineage-file $Q/meshes_slotted_C/slotted_mesh_report.json --lineage-file $J/code/res25.py
