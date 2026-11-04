"""G10 bundle builder (GT-free; spoon only). New bundle = copy of selected20_final_v6; for each listed episode the wooden_spoon
pose is rolled about its long (PCA) axis through the mesh centroid by the ball-Viterbi roll track (balltrack.py output),
unwrapped and smoothed with a 5-frame moving average where it changes (no instantaneous flips). Meshes, other objects,
cameras untouched. Never overwrites.
usage: build_v8.py OUT_BUNDLE EP:TAG [EP:TAG ...]"""
import sys, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import C, RR, roll_axis, loadmesh
from scipy.spatial.transform import Rotation as Rot
OUT = Path(sys.argv[1]); V6 = Path(RR) / 'selected20_final_v6'
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
shutil.copytree(V6, OUT)
rep = dict(base=str(V6), rule='spoon roll = ball-Viterbi track (streamG10_mp/code/balltrack.py), smoothed 5 frames', episodes={})
for arg in sys.argv[2:]:
    ep, tag = arg.split(':'); ep = int(ep)
    roll = np.load(f'{C}/streamG10_mp/out/balltrack_ep{ep}_{tag}_roll.npy').astype(float)
    js = json.load(open(f'{C}/streamG10_mp/out/balltrack_ep{ep}_{tag}.json'))
    fol = OUT / f'episode_{ep:06d}'
    d = dict(np.load(fol / 'object_poses.npz')); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
    v, f = loadmesh(fol / 'meshes/wooden_spoon.obj', 1 << 30)
    R = np.load(f'{C}/streamG10_mp/out/rollscan_ep{ep}.npz'); c, ax = R['centroid'], R['axis']
    un = np.degrees(np.unwrap(np.radians(roll)))
    k = 2; sm = un.copy()
    for t in range(len(un)):
        a, b = max(0, t - k), min(len(un), t + k + 1); sm[t] = un[a:b].mean()
    sm[0] = 0.0
    cam = d['camera_from_object'][:, s].copy()
    for t in range(len(cam)):
        if abs(sm[t]) < 1e-6: continue
        Rr = Rot.from_rotvec(np.radians(sm[t]) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = Rr; X[:3, 3] = c - Rr @ c
        cam[t] = cam[t] @ X
    d['camera_from_object'] = d['camera_from_object'].copy(); d['camera_from_object'][:, s] = cam
    d['world_from_object'] = d['world_from_object'].copy(); d['world_from_object'][:, s] = d['world_from_rgb'] @ cam
    np.savez(fol / 'object_poses.npz', **d)
    info = dict(tag=tag, segments=js['segments'], flipped_frac=js['flipped_frac'], offset=js['offset'], roll_smoothed=np.round(sm, 2).tolist(),
                proxy_v24=js['v24'], proxy_v25=js['v25'])
    (fol / 'g10_spoon_roll.json').write_text(json.dumps(info, indent=1)); rep['episodes'][ep] = {k: info[k] for k in ('tag', 'segments', 'flipped_frac')}
    print(ep, tag, js['segments'])
(OUT / 'g10_report.json').write_text(json.dumps(rep, indent=1))
