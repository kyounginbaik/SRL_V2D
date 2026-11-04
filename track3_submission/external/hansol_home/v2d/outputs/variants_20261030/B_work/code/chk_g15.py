import sys, json, numpy as np
from pathlib import Path
G = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15')
sys.path.insert(0, str(G / 'code')); from retime import shift
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
D = json.loads((G / 'out/sync_delta_t.json').read_text())
print('keys', sorted(D)[:5], len(D))
V8, V14 = RR / 'selected20_final_v8_g10b', RR / 'selected20_final_v14_g17'
mx = lambda a: float(np.abs(a).max())
for ep in sorted(V14.glob('episode_*')):
    e = int(ep.name[-6:]); a = np.load(V8 / ep.name / 'object_poses.npz'); b = np.load(ep / 'object_poses.npz')
    d = D[f'evaluation_{e}']; cam = np.load(ep / 'camera.npz')['world_from_camera'][:len(a['world_from_rgb'])]
    camold = np.load(ep / 'camera_pre_g15.npz')['world_from_camera'][:len(a['world_from_rgb'])]
    names = [str(x) for x in b['names']]
    r = dict(keys=sorted(b.files) == sorted(a.files), names=names == [str(x) for x in a['names']], F=(len(a['world_from_rgb']), len(b['world_from_rgb'])),
             wfr=mx(shift(cam, d) - b['world_from_rgb']), camold_vs_v8wfr=mx(camold - a['world_from_rgb']))
    for k, n in enumerate(names):
        Wp = shift(np.einsum('tij,tjk->tik', cam, a['camera_from_object'][:, k]), d)
        Cp = shift(a['camera_from_object'][:, k].astype(float), d)
        r[n] = (round(mx(Wp - b['world_from_object'][:, k]), 9), round(mx(Cp - b['camera_from_object'][:, k]), 9))
    incons = mx(np.einsum('tij,tnjk->tnik', b['world_from_rgb'], b['camera_from_object']) - b['world_from_object'])
    r['v14_rel_incons'] = round(incons, 6)
    oth = {k: bool(np.array_equal(a[k], b[k])) for k in a.files if k not in ('world_from_object', 'camera_from_object', 'world_from_rgb')}
    r['other_keys_equal'] = oth
    print(e, d if np.ndim(d) == 0 else sorted(set(np.asarray(d).tolist())), r)
# public
C = RR / 'canonical'
for split in ['public', 'public_heldout']:
    for root in sorted((C / split).glob('episode_*')):
        e = int(root.name[-6:]); g = root / 'exports' / 'g15_camrt'; r8 = root / 'exports' / 'r8_base'
        if not g.exists(): print(split, e, 'no g15_camrt'); continue
        a = np.load(r8 / 'object_poses.npz'); b = np.load(g / 'object_poses.npz')
        key = [k for k in D if k.endswith(f'_{e}') and not k.startswith('evaluation')]
        cp = G / 'cam' / f'public_{e}' / 'camera.npz'
        cam = np.load(cp)['world_from_camera'][:len(a['world_from_rgb'])]
        d = D[key[0]] if key else None
        rj = json.loads((g / 'retime.json').read_text()) if (g / 'retime.json').exists() else None
        Wp = shift(np.einsum('tij,tnjk->tnik', cam, a['camera_from_object']).reshape(-1, 4, 4), 0) if False else None
        errs = [mx(shift(cam @ a['camera_from_object'][:, k], d) - b['world_from_object'][:, k]) for k in range(len(a['names']))]
        print(split, e, key, rj and rj.get('delta_frames'), 'Werr', max(errs), sorted(f.name for f in g.iterdir()))
