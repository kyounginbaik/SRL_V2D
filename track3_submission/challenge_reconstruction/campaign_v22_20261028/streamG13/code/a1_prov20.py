"""Provenance + kit-faithful pairwise errors among ep20 tracks."""
import sys, json, numpy as np
from tracks import *; import kit
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
S = {'v10': target('v10', ep), 'c21': target('c21', ep), 'c27': target('c27', ep)}
if ep in (14, 15, 16, 20, 25, 28, 30): S['fro'] = bundle(FRO, ep)
else: S['fro'] = bundle(f'{R}/final_recon', ep)   # 'fro' = old final_recon for the earlier-13 episodes
for p in ['first', 'v13', 'v21C', 'v22b', 'v24', 'v26']:
    S[p + '_ach'] = package(p, ep); S[p + '_ref'] = package(p, ep, 'reference')
# provenance: is each package's reference identical (kit-aligned) to a known target?
W = [(0, 10), (10, 50), (50, 110), (110, 150), (150, 202)]
res = {}
def show(a, b):
    pos, ori = kit.errors(S[a], S[b], ep)
    names = [x[0] for x in S[a]]
    s = f'{a:>9s} vs {b:<5s}: ' + '  '.join(f'{n[:6]} {pos[:, i].mean():5.2f}/{ori[:, i].mean():5.1f}' for i, n in enumerate(names))
    s += ' | win pos ' + ' '.join('[' + ','.join(f'{pos[x:y, i].mean():.1f}' for i in range(len(names))) + ']' for x, y in W)
    print(s, flush=True); res[f'{a}|{b}'] = dict(pos=pos.mean(0).tolist(), ori=ori.mean(0).tolist(), win=[pos[x:y].mean(0).tolist() for x, y in W])
for p in ['first', 'v13', 'v21C', 'v22b', 'v24', 'v26']:
    for t in ['fro', 'c21', 'v10']:
        if t in S: show(p + '_ref', t)
for t in ['fro', 'c21', 'c27']: show('v10', t)
if 'fro' in S: show('fro', 'c21'); show('fro', 'c27')
for p in ['first', 'v21C', 'v22b', 'v26']:
    for t in ['v10', 'fro', 'c21']:
        if t in S: show(p + '_ach', t)
json.dump(res, open(f'../out/a1_prov{ep}.json', 'w'), indent=1)
