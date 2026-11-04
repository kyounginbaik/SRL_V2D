#!/bin/bash
# usage: score.sh EP TAG_DIR LABEL  -> streamT_C356/scores/LABEL/score.json (vs compose_target, official anchor procedure)
set -euo pipefail
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; T=$C/streamT_C356; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
O=$T/scores/$3; D=$(realpath $2)
cd /tmp && $PY -B $T/code/make_sub.py $1 $D $O >/dev/null && $PY -B $C/compose_target/score_vs_compose.py $O/submission.parquet $O/results $C/compose_target $O/score.json $1 2>&1 | grep -v Warn
