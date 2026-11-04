#!/bin/bash
# usage: ilcsum.sh JOBID...   (summarise mres ilc/cem IT lines)
U=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamU8_15_16
for j in "$@"; do echo "== $j $(grep -o '"out": "[^"]*"' $U/logs/res-$j.log | head -1)"; grep -E '^(INIT|IT|FINAL|RESULT)' $U/logs/res-$j.log | python3 -c "
import json,sys
for l in sys.stdin:
  t,_,b=l.partition(' ')
  try: r=json.loads(b)
  except Exception: print(l[:300].strip()); continue
  if t=='IT' and 'best_candidate' in r: print(r['iteration'], r['best_candidate'], [round(x,3) for v in r['objects'].values() for x in v], round(r['cost'],2), round(r['prev_cost'],2))
  elif t=='IT': print(r['iteration'], [round(x,3) for v in r['best']['objects'].values() for x in v], round(r['best']['cost'],2), r.get('median', r.get('n_margin')))
  else: print(t, json.dumps(r)[:300])
"; done
