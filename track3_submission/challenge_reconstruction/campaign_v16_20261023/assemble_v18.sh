#!/bin/bash
# v16 baseline: the v15 actors re-executed under evaluator build v2, plus the legacy ep37 actor (user decision 2026-10-23).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $C/package_code/assemble_twenty_v2.py \
  --build-manifest /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/build_manifest.json \
  --common-evaluator /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/package_code/common_evaluator.py \
  $(cat $C/assemble_v18.args | tr -d '\\') \
  --output /data1/SRL_V2D/challenge_submissions/track3_v18_20261023/all_20_episodes \
  --reason "v17 plus phase-3 components on reconstruction meshes: episode 1 lid-on corrected reference actor; episode 10 re-planned on our reconstruction cups (GT cups removed); episode 30 re-planned on our reconstruction dish rack (scan rack removed); episode 32 scoop-retention actor; episode 33 re-timed left release; episode 35 position-ILC margin actor; all other episodes unchanged from v17 (evaluator build v2)"
