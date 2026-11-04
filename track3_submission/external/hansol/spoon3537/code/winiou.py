import json
for e in (36, 37, 35):
    w = json.load(open(f'/data1/hansol/episode_review/out/ep{e}/diag_windows.json'))
    out = []
    for x in w:
        s = x['wooden_spoon']
        if s.get('iouR') is None: continue
        out.append('%.1f:%.2f/%d/%d' % (x['t0'], s['iouR'], s.get('rw_max', 0), s.get('rv', 0)))
    print(e, ' '.join(out))
