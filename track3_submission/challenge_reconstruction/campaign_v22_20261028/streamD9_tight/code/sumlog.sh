#!/bin/bash
# usage: sumlog.sh LOG  -> per-window best and FINAL lines
f=$1; grep -h '^INIT' $f | cut -c1-200
grep '^WIN' $f | python3 -c "
import sys,json
for l in sys.stdin:
    r=json.loads(l[4:]); b=r['best'][0]; s=b['s']
    print(r['window'], 'n',r['n'], 'zero', round(min(r['zero']),1), 'best', round(b['cum'],1), ' '.join(f'{k}:{v[0]:.2f}/{v[1]:.1f}' if isinstance(v,list) else f'{k}:{v:.2f}' for k,v in s.items()), b['last'][b['last'].find('{'):][:60], r['elapsed_s'])"
grep -h '^FINAL' $f
