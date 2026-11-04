#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=v2-cand
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/logs/%x-%j.log
# usage: sbatch run_one.sbatch EP COMPONENT_DIR TAG   -> output common_v2_candidates/<TAG>/episode_XXXXXX
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EP=$1; COMP=$2; TAG=$3
B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
V2=/data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023
KIT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit
NAME=$(printf episode_%06d $EP)
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $V2/package_code/common_evaluator.py evaluate --backend legacy --episode $EP --legacy-component $COMP --build-manifest $V2/build_manifest.json --kit $KIT --output $B/$TAG/$NAME
