import json, sys
for d in sys.argv[1:]:
    a = json.load(open(d + '/attempts.json'))
    for r in (a if isinstance(a, list) else a.get('attempts', [])):
        oe = r.get('object_errors', {}); print(d.split('/')[-1], r['checkpoint'].split('/')[-3], r.get('passed'), ' '.join(f"{k[:5]}:{v['cm']:.2f}/{v['deg']:.1f}" for k, v in oe.items()))
