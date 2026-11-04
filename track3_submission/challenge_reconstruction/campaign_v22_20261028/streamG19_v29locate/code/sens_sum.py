import json, glob, os
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
rows = {}
for f in sorted(glob.glob(f'{S}/out/sens/ep*.json'), key=lambda x: int(x.split('ep')[-1][:-5])):
    d = json.load(open(f)); r = {}
    for k, v in d['rot'].items():
        n = k.split('/')[0]; a = v['fail_at_deg']
        if a is not None and (n not in r or a < r[n][0]): r[n] = (a, k.split('/')[1], v['errors'])
    t = {}
    for k, v in d['trans'].items():
        n = k.split('/')[0]; a = v['fail_at_cm']
        if a is not None and (n not in t or a < t[n][0]): t[n] = (a, k.split('/')[1])
    rows[d['ep']] = dict(base=d['base'][1], names=d['names'], min_rot=r, min_trans=t)
    print(d['ep'], d['names'], 'base', d['base'][1], '| min body rot to fail:', {n: x[:2] for n, x in r.items()}, '| min trans:', {n: x for n, x in t.items()})
json.dump(rows, open(f'{S}/out/sens_summary.json', 'w'), indent=1)
