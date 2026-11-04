"""Per episode: own-target (OUR provenance, no COMPOSE) exact build-v2 candidates ranked by knockout strength
q = min over truths (scan-in-v14, scan-in-v9, c21, c27, fro) of max_obj max(pos/3cm, ori/30deg); q > 1 = fails under every truth."""
import json, os, sys
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
P = json.load(open(f'{S}/out/provenance.json'))
TR = ['scanv14', 'scanv9', 'c21', 'c27', 'fro']
res = {}
for ep in [20, 33, 3, 1, 19, 35, 36, 5, 6, 8, 10, 14, 15, 16, 25, 28, 30, 32, 37, 38]:
    rows = []
    for k, v in P.items():
        if v['ep'] != ep or not v['exact']: continue
        own = 'OUR' in v['cls'] and 'COMPOSE' not in v['cls']
        q = {}
        for t in TR:
            fn = f'{S}/out/tab/K_{t}/ep{ep}.json'
            d = json.load(open(fn)).get(v['tag']) if os.path.exists(fn) else None
            q[t] = None if not d or 'obj' not in d else max(max(p / 3, o / 30) for p, o in d['obj'])
        qs = [x for x in q.values() if x is not None]
        rows.append((min(qs) if qs else -1, own, v['tag'], {t: (round(x, 2) if x else None) for t, x in q.items()}, v['cls'], v['sources'][:2]))
    rows.sort(key=lambda r: (-r[1], -r[0]))
    res[ep] = rows
    print(f'=== ep{ep}')
    for r in rows[:6]: print('  q=%.2f own=%s %s %s %s' % (r[0], r[1], r[2], r[3], [s.split('challenge_reconstruction/')[-1][:70] for s in r[5]]))
json.dump({str(k): v for k, v in res.items()}, open(f'{S}/out/ko_single_pick.json', 'w'), indent=1)
