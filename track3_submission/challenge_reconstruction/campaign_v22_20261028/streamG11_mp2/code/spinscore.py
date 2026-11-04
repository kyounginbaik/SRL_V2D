"""SCORER-SIDE validation only (public GT via bscore): kit error of a public export when OBJ's whole track gets a constant
body-frame spin delta about its thin (knob) axis through the centroid (same definition as spinscan.py).
Optional --perframe NPZ (a spinscan output or any npz with 'frames' and 'delta_per_frame') applies a per-frame delta.
usage: spinscore.py EXPORT_DIR OBJ [--step 5]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/boards')
import bscore
ap = argparse.ArgumentParser(); ap.add_argument('folder', type=Path); ap.add_argument('obj'); ap.add_argument('--step', type=float, default=5)
ap.add_argument('--lo', type=float, default=-60); ap.add_argument('--hi', type=float, default=61)
a = ap.parse_args(); folder = a.folder; root = folder.parents[1]
rec = json.loads((root / 'prepared.json').read_text())['record']; ep = rec['episode']
d = np.load(folder / 'object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index(a.obj)
mj = json.loads((folder / 'meshes.json').read_text()) if (folder / 'meshes.json').exists() else {}
mp = {n: mj.get(n, str(root / 'objects' / n / 'mesh.obj')) for n in names}
v = np.asarray(trimesh.load(mp[a.obj], force='mesh', process=False).vertices); c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); ax = U[:, 0]
h = (v - c) @ ax; ax = ax * np.sign(np.mean(h ** 3))
res = {}
for dl in np.arange(a.lo, a.hi, a.step):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(np.deg2rad(dl) * ax).as_matrix(); T[:3, 3] = c - T[:3, :3] @ c
    W = d['world_from_object'].copy(); W[:, s] = W[:, s] @ T
    e, _ = bscore.score(ep, {n: (W[:, i], mp[n]) for i, n in enumerate(names)}, metrics=False)
    res[float(dl)] = e[a.obj]; print(ep, a.obj, dl, [round(x, 2) for x in e[a.obj]], flush=True)
print('JSON', json.dumps(dict(ep=ep, obj=a.obj, res=res)))
