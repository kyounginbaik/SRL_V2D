#!/bin/bash
# v16 baseline: the v15 actors re-executed under evaluator build v2, plus the legacy ep37 actor (user decision 2026-10-23).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $C/package_code/assemble_twenty_v2.py \
  --build-manifest /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/build_manifest.json \
  --common-evaluator /data1/SRL_V2D/challenge_reconstruction/evaluator_build_v2_20261023/package_code/common_evaluator.py \
  $(cat $C/assemble_v21.args | tr -d '\\') \
  --output /data1/SRL_V2D/challenge_submissions/track3_v21_20261024/all_20_episodes \
  --reason "v20 plus: episodes 15 and 16 spoon re-targeted to the refined spoon track (mocap-ball and silhouette evidence); episode 32 spoon roll corrected in three windows and followed with a power grasp; episode 33 spoon roll corrected at 6.7-8.25 s and followed by the robot; episode 25 board-2 position corrected where stereo supports it (same robot actions); all other episodes unchanged from v20 (evaluator build v2)"
