#!/bin/bash
# usage: score_traj.sh TRAJ_NPZ OUT_DIR  -> OUT_DIR/score.json (pre-screen vs compose_target)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32; CT=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target
$PY $S/code/make_sub32_traj.py $1 $2 && cd /tmp && $PY $CT/score_vs_compose.py $2/submission.parquet $2/results $CT $2/score.json 32 2>&1 | grep "^32" | cut -c1-300
