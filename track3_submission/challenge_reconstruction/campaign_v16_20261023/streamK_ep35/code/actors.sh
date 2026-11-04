#!/bin/bash
# list distilled attempts. usage: actors.sh
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
for d in $K/distilled/*; do [ -f $d/attempts.json ] || continue; echo "## $(basename $d)"; python3 -c "
import json; a=json.load(open('$d/attempts.json'))
for r in a: print('  ', r['checkpoint'].split('/')[-3], 'PASS' if r['passed'] else 'fail', r.get('object_errors',{}).get('wooden_spoon'), r.get('error','')[:60])
"; done
