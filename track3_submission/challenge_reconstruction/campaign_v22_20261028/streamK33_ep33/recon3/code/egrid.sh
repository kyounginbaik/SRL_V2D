#!/bin/bash
R=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
SUB=/data1/SRL_V2D/challenge_submissions_SRL; O=$R/out/egrid; mkdir -p $O
for lab in $(ls $R/tmp_egrid); do for P in track3_v29_20261030 track3_v26_20261029 track3_v31a_HGA_20261101 ep33_fix_20261102/K33_H1rest_b2 ep33_fix_20261102/K33_H7ball_b5; do
  n=$(echo $P | tr / _); echo "$PY -B -u $S $SUB/$P/all_20_episodes/track3_submission.parquet $SUB/$P/all_20_episodes/results $R/tmp_egrid/$lab $O/${lab}__$n.json 33 > $O/${lab}__$n.log 2>&1"
done; done | xargs -P 8 -I{} sh -c "{}"
echo DONE
