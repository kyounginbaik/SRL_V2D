import json, sys
for d in sys.argv[1:]:
    a = json.load(open(d + '/attempts.json')); a = a if isinstance(a, list) else a.get('attempts', a)
    for r in sorted(a, key=lambda r: r.get('maximum_ratio', 9)):
        oe = r.get('object_errors', {}); print(d.split('/')[-1][:40], r['checkpoint'].split('/')[-3], r.get('passed'), round(r.get('maximum_ratio', 9), 3), {k[-8:]: (v['cm'], v['deg']) for k, v in oe.items()})
