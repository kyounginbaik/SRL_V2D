#!/bin/bash
# usage: score_run.sh EP RUN_DIR LABEL [tl] -> streamN8/scores/LABEL/score.json (raw run vs target_final_v5 with v22b package meshes)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/code/env.sh
O=$SN/scores/$3; cd /tmp && $PY -B $SN/code/score_run.py $1 $2 $O 2>&1 | grep -v Warn; $PY -B $C/compose_target/score_vs_compose.py $O/submission.parquet $O/results $V22/target_final_v5 $O/score.json $1 2>&1 | grep -v Warn | tail -3
if [ "$4" == tl ]; then $PY -B $SN/code/timeline_v5.py $1 $O/submission.parquet $O/results/episode_$(printf %06d $1).parquet "" 15 $O/tl.npz 2>&1 | grep -v Warn; fi
