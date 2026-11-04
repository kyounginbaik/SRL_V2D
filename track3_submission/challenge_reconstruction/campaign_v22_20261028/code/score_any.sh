#!/bin/bash
# usage: score_any.sh EP TAG_DIR LABEL [TARGET] -> campaign_v22_20261028/scores/cand/LABEL/score.json (default TARGET target_final_v5)
set -euo pipefail
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
T=${4:-$C/target_final_v5}; O=$C/scores/cand/$3; D=$(realpath $2)
cd /tmp && OMP_NUM_THREADS=1 $PY -B $C/code/make_sub_any.py $1 $D $O >/dev/null && OMP_NUM_THREADS=1 $PY -B /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py $O/submission.parquet $O/results $T $O/score.json $1 2>&1 | grep -v Warn
