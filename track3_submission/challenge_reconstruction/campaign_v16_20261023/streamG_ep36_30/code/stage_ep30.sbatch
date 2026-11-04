#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=G30-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/stage30-%j.log
# usage: sbatch stage_ep30.sbatch ACTOR_REPORT_JSON BASE_DIR [extra --lineage-file ...]
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021
AR=$1; BASE=$2; shift 2
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u $RS/ep30_v8/code/stage_seven_component_corrected_track.py --episode 30 --base $BASE --actor-report $AR --lineage-file $RS/ep30_v8/collision_recon/episode_000030/sweeper_track_correction.json "$@"
