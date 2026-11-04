"""K33 beam progress: python3 prog.py LOG"""
import json, sys
for l in open(sys.argv[1]):
    if l.startswith('INIT'): print(l[:240].strip())
    if l.startswith('WIN '):
        w = json.loads(l[4:]); b = w['best'][0]
        print(w['window'], w['elapsed_s'], {k: ([round(x, 2) for x in v] if isinstance(v, list) else round(v, 2)) for k, v in b['s'].items()})
    if l.startswith('FINAL '): print(l[:300].strip())
