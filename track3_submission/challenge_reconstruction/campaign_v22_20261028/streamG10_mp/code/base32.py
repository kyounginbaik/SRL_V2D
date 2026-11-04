"""v24 / v25 rollouts of one episode vs existing reconstructions; per-frame errors saved.
usage: base32.py EP"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
TG = {'v6': f'{C}/target_final_v6', 'v7rot': f'{C}/target_final_v7_g9rot', 'v7g9': f'{C}/target_final_v7_g9',
      'c27': '/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27',
      'c21': '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target'}
out = {}; pf = {}
for pk in ['v24', 'v25']:
    names, subT, cands = load_sub(PKG[pk], EP)
    for tl, td in TG.items():
        try:
            refT, rm = load_target(td, EP, names)
        except Exception as e:
            print(pk, tl, 'missing', e); continue
        S = Scorer(EP, names, cands, rm)
        s, pos, ori = S.summary(subT, refT)
        out[f'{pk}_{tl}'] = s; pf[f'{pk}_{tl}_pos'] = pos; pf[f'{pk}_{tl}_ori'] = ori
        print(EP, pk, tl, json.dumps(s), flush=True)
json.dump(out, open(f'{C}/streamG10_mp/out/base{EP}.json', 'w'), indent=1)
np.savez(f'{C}/streamG10_mp/out/base{EP}_pf.npz', **pf)
