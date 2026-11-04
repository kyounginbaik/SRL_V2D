import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/code')
from elib import *
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/C_ep30_f48/episode_000030'
from pathlib import Path
names = ['mini_sweeper', 'plastic_dish_rack']
ach, ref = results_tracks(next(Path(C, 'results').glob('*.parquet')))
for slot, n in enumerate(names):
    mp = next(Path(C, 'meshes').rglob(f'{n}.obj'))
    X, ca, rot, anchor = registration(30, slot, n, mp)
    print(n, 'reg rot deg', np.degrees(np.linalg.norm(R.from_matrix(rot).as_rotvec())), 'ca', ca, 'anchor', anchor)
    for s in [1, 2, 3, 7, 11]:
        X2, *_ = registration(30, slot, n, mp, seed=s)
        print('   seed', s, 'X vs X2 deg', np.degrees(np.linalg.norm(R.from_matrix(X[:3,:3].T@X2[:3,:3]).as_rotvec())))
    T = target(30, n) @ X
    globals()['T_' + n[:4]] = T; globals()['R_' + n[:4]] = ref[n]
# relative pose rack in sweeper frame at f0: target vs our physics reference
for f in [0, 40, 52, 60, 120, 262]:
    a = np.linalg.inv(T_mini[f]) @ T_plas[f]; b = np.linalg.inv(R_mini[f]) @ R_plas[f]
    print('f', f, 'rel sweeper->rack diff (cm, deg)', [x[0] for x in errs(a[None], b[None])])
G0 = R_mini[0] @ np.linalg.inv(T_mini[0])
for n, T, Rr in [('sweeper', T_mini, R_mini), ('rack', T_plas, R_plas)]:
    p, d = errs(np.einsum('ij,njk->nik', G0, T), Rr)
    print(n, 'after G0 vs our phys ref: mean', p.mean(), d.mean(), 'per 30f', np.round(p[::30],1), np.round(d[::30]))
# rack-anchored instead
G0r = R_plas[0] @ np.linalg.inv(T_plas[0])
for n, T, Rr in [('sweeper', T_mini, R_mini), ('rack', T_plas, R_plas)]:
    p, d = errs(np.einsum('ij,njk->nik', G0r, T), Rr)
    print(n, 'rack-anchored vs our phys ref: mean', p.mean(), d.mean(), 'per 30f', np.round(p[::30],1), np.round(d[::30]))
