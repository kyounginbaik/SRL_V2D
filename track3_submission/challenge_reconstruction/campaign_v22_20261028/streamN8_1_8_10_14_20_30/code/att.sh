#!/bin/bash
# usage: att.sh DISTILLED_DIR... : list actors (cm/deg per object) sorted by max ratio
for d in "$@"; do python3 - "$d" <<'P'
import json,sys
d=sys.argv[1]; a=json.load(open(d+'/attempts.json')); a=a if isinstance(a,list) else a.get('attempts',a)
rows=sorted(a,key=lambda r:r.get('maximum_ratio',9))
print('==',d.split('streamN8_1_8_10_14_20_30/')[-1])
for r in rows: print('  ',r['checkpoint'].split('/')[-3], r.get('passed'), ' '.join(f"{k}:{v['cm']:.2f}/{v['deg']:.1f}" for k,v in r.get('object_errors',{}).items()))
P
done
