#!/bin/bash
# v16 baseline: the v15 actors re-executed under evaluator build v2, plus the legacy ep37 actor (user decision 2026-10-23).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $C/package_code/assemble_twenty_v2.py \
  --build-manifest /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/build_manifest.json \
  --common-evaluator /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/package_code/common_evaluator.py \
  $(cat $C/assemble_v16_baseline.args | tr -d '\\') \
  --output /data1/SRL_V2D/challenge_submissions/track3_v16baseline_20261023/all_20_episodes \
  --reason "evaluator build v2 (legacy backend allowed for episode 37); episode 37 switched to the legacy learned actor; all other episodes are the v15 actors freshly re-executed"
