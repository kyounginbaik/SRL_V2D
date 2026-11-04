#!/bin/bash
# usage: retro.sh TARGET_LABEL -> kit-faithful ep33 score of every officially FAILING ep33 rollout vs targets/TARGET_LABEL
T=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/targets/$1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
O=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/recon3/out/retro/$1; mkdir -p $O
SUB=/data1/SRL_V2D/challenge_submissions_SRL
for P in track3_v29_20261030 track3_v26_20261029 track3_v31a_HGA_20261101 ep33_fix_20261102/K33_H1rest_b2 ep33_fix_20261102/K33_H7ball_b5; do
  n=$(echo $P | tr / _); echo "$PY -B -u $S $SUB/$P/all_20_episodes/track3_submission.parquet $SUB/$P/all_20_episodes/results $T $O/$n.json 33 > $O/$n.log 2>&1"
done | xargs -P 5 -I{} sh -c "{}"
for f in $O/*.json; do echo "$(basename $f .json) $(python3 -c "import json;d=json.load(open('$f'))['33'];print(d['objects'],d['scores']['MP-SR'])")"; done
