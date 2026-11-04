#!/bin/bash
# usage: score_run.sh EP RUN_DIR LABEL -> streamD_ep1_20/scores/LABEL/score.json (+ per-frame timeline when 4th arg = tl)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/env.sh
O=$SD/scores/$3; cd /tmp && $PY -B $SD/code/score_run.py $1 $2 $O 2>&1 | grep -v Warn; $PY -B $C/compose_target/score_vs_compose.py $O/submission.parquet $O/results $V22/target_final_v5 $O/score.json $1 2>&1 | grep -v Warn
if [ "$4" == tl ]; then $PY -B $SD/code/timeline_v5.py $1 $O/submission.parquet $O/results/episode_$(printf %06d $1).parquet "" 20 $O/tl.npz 2>&1 | grep -v Warn; fi
