#!/bin/bash
# usage: tl.sh EP LABEL [BIN]  (LABEL = scores/cand/<LABEL> produced by score_any.sh)
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /tmp && OMP_NUM_THREADS=2 $PY -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/timeline_v5.py $1 $C/scores/cand/$2/submission.parquet $C/scores/cand/$2/results/episode_$(printf %06d $1).parquet "" ${3:-15} /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/diag/tl_$2.npz 2>&1 | grep -v Warn
