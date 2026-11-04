#!/bin/bash
# usage: score_grid.sh EP DISTILLED_DIR PREFIX  -> one line per actor (scorer vs target_final_v5)
D=$(realpath $2); for a in $D/actor_*; do [ -f $a/evaluation/result.json ] || continue; n=$3_$(basename $a)
  /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD_ep1_20/code/score_run.sh $1 $a/evaluation $n >/dev/null 2>&1
  echo "$(basename $a) $(python3 -c "import json;d=json.load(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD_ep1_20/scores/$n/score.json'))['$1'];print(d['scores']['MP-SR'], d['objects'])")"; done
