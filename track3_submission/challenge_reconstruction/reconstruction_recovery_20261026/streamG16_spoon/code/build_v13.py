"""G16 bundle builder (GT-free; spoon only). New bundle = copy of selected20_final_v12b_g15; for each listed episode the
wooden_spoon pose gets ONE constant body-frame roll about its long PCA axis through the surface centroid (same axis and
sign convention as g16scan / G10 rollscan), applied to world_from_object and camera_from_object on every frame.
Meshes, other objects, camera untouched. Never overwrites.
usage: build_v13.py OUT_BUNDLE EP:ROLL_DEG [EP:ROLL_DEG ...]"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis, loadmesh
from scipy.spatial.transform import Rotation as Rot
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
OUT = Path(sys.argv[1]); BASE = RR / 'selected20_final_v12b_g15'
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
shutil.copytree(BASE, OUT)
rep = dict(base=str(BASE), ground_truth_used=False,
           rule='G16: constant spoon body roll where every early-window (f0-23) GT-free estimator agrees (2D ball roll scan plain/'
                'image-shift, 3D stereo ball plain/shift, 6-DoF ball fit 2D/3D): |median| >= 10 deg, same sign, bootstrap CI excludes 0',
           evidence='reconstruction_recovery_20261026/streamG16_spoon/out/{roll_offsets,offs2_ep*,offs3_ep*,f0fit_ep*_w24}.json',
           episodes={})
for arg in sys.argv[2:]:
    ep, roll = arg.split(':'); ep = int(ep); roll = float(roll)
    fol = OUT / f'episode_{ep:06d}'
    d = dict(np.load(fol / 'object_poses.npz')); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
    v, f = loadmesh(fol / 'meshes/wooden_spoon.obj', 1 << 30)
    c, ax = roll_axis(v, f)
    R = Rot.from_rotvec(np.radians(roll) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = R; X[:3, 3] = c - R @ c
    for k in ('world_from_object', 'camera_from_object'):
        a = d[k].copy(); a[:, s] = (a[:, s].astype(np.float64) @ X).astype(a.dtype); d[k] = a
    np.savez_compressed(fol / 'object_poses.npz', **d)
    info = dict(roll_deg=roll, axis_body=ax.tolist(), centroid_body=c.tolist(), md5=hashlib.md5((fol / 'object_poses.npz').read_bytes()).hexdigest()[:8])
    (fol / 'g16_spoon_roll.json').write_text(json.dumps(info, indent=1)); rep['episodes'][ep] = info
    print(ep, info)
(OUT / 'G16_CHANGES.json').write_text(json.dumps(rep, indent=1))
