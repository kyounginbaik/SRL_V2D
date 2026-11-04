import sys, json
for f in sys.argv[1:]:
    rows = []
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        if 'objects' not in d: continue
        o = d['objects']; rows.append((max(max(v[0] / 3, v[1] / 30) for v in o.values()), d['out'].split('streamF_margins/')[1], d['mp_sr'], o))
    for r in sorted(rows): print('%.2f' % r[0], r[1], r[2], ' '.join('%s %.2f/%.1f' % (k[:6], v[0], v[1]) for k, v in r[3].items()))
