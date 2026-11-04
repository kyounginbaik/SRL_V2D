#!/bin/bash
# [spoon3537 copy of campaign_v16 common_v2_candidates/run_one.sbatch body: output -> /data1/hansol/spoon3537/rt/candidates/<TAG>/<episode>]
# usage: s3run_one.sh EP COMPONENT_DIR TAG
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
EP=$1; COMP=$2; TAG=$3; B=/data1/hansol/spoon3537/rt/candidates
V2=/data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023
KIT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit
NAME=$(printf episode_%06d $EP); mkdir -p $B/$TAG
cd /tmp && /home/hansol/v2d/.venvs/v2d-host/bin/python -B -u $V2/package_code/common_evaluator.py evaluate --backend legacy --episode $EP --legacy-component $COMP --build-manifest $V2/build_manifest.json --kit $KIT --output $B/$TAG/$NAME
