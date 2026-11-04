#!/bin/bash
# H13a progress: per beam log the number of windows done, last window, elapsed, best summary; FINAL/VS lines
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/h13a/logs
for f in H13a-b*.log; do python3 - $f <<'PY'
import sys, json
f = sys.argv[1]; w = None; n = 0; fin = []; init = None
for l in open(f):
    if l.startswith('WIN '): n += 1; w = json.loads(l[4:])
    elif l.startswith('FINAL '): fin.append(json.loads(l[6:]))
    elif l.startswith('INIT '): init = l[5:220].strip()
b = w and w['best'][0]; s = b and {k: v for k, v in b.items() if k in ('s', 'sp', 'cup')}
print(f, 'wins', n, w and w['window'], w and w['elapsed_s'], json.dumps(s)[:220] if s else init)
for r in fin[:2]: print('   FINAL', json.dumps(r)[:260])
PY
done
grep -h "^VS" H13a-svst*.log
squeue -u exaFLOPs -o "%.8i %.14j %.8M %R" | grep H13a
