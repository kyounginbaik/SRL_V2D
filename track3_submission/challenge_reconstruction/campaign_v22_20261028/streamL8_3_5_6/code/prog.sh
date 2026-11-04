#!/bin/bash
# summarise ILC / CEM / release logs: prog.sh JOBID...
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6
for j in "$@"; do echo "== $j $(grep -o 'out [^ ]*\|--out [^ ]*' logs/res-$j.log | head -1) $(sed -n 's/.*"out": "\([^"]*\)".*/\1/p' logs/res-$j.log | head -1)"
grep -E "^IT|Trace|Error|FINAL|RESULT|PREFIX|BASE" logs/res-$j.log | python3 -c "
import sys,json
for l in sys.stdin:
  if not l.startswith('IT'): print(l.strip()[:400]); continue
  s=l[3:].strip()
  if s[0].isdigit(): s=s.split(' ',1)[1]
  d=json.loads(s); b=d.get('best',d); o=b.get('objects') or {}
  print(d.get('iteration'), d.get('best_candidate',''), round(b['cost'],3), {k:(round(v[0],2),round(v[1],1)) for k,v in o.items()}, d.get('n_better',d.get('n_margin','')))"; done
