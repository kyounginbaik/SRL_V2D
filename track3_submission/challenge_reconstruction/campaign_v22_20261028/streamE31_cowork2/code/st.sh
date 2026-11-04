#!/bin/bash
# E31 progress: per beam log windows done, last window, elapsed, best summary; FINAL / VS lines
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2/logs
for f in E31-b*.log; do python3 - $f <<'PY'
import sys, json
f = sys.argv[1]; w = None; n = 0; fin = []; init = None
for l in open(f):
    if l.startswith('WIN '): n += 1; w = json.loads(l[4:])
    elif l.startswith('FINAL '): fin.append(json.loads(l[6:]))
    elif l.startswith('INIT '): init = l[5:200].strip()
b = w and w['best'][0]; s = b and {k: [round(x, 2) for x in v] for k, v in (b.get('s') or b.get('objects') or {}).items() if k != 'mppe'}
print(f, 'wins', n, w and w.get('window'), w and round(w.get('elapsed_s', 0)), json.dumps(s) if s else init, b and round(b.get('cum', 0)))
for r in fin[:3]: print('   FINAL', json.dumps(r)[:240])
PY
done
grep -h "^VS" E31-svst*.log 2>/dev/null
squeue -u exaFLOPs -o "%.8i %.12j %.8M %R" | grep E31
