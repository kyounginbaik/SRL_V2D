"""Scoring-only scan: Gaussian temporal smoothing of world tracks (translation and rotation, sigma in frames) on an export.
Usage smooth_scan.py EP TAG"""
import sys, json, numpy as np, kitx
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
ep, tag = int(sys.argv[1]), sys.argv[2]; X = kitx.load_export(ep, tag)
def smooth(T, st, sr):
    out = T.copy()
    if st > 0: out[:, :3, 3] = gaussian_filter1d(T[:, :3, 3], st, axis=0, mode='nearest')
    if sr > 0:
        q = Rot.from_matrix(T[:, :3, :3]).as_quat()
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
        q = gaussian_filter1d(q, sr, axis=0, mode='nearest'); q /= np.linalg.norm(q, axis=1, keepdims=True)
        out[:, :3, :3] = Rot.from_quat(q).as_matrix()
    return out
base = kitx.score(ep, {n: X['W'][:, i] for i, n in enumerate(X['names'])}, X['meshes'])
regs = {n: (base['cas'][s], base['rots'][s]) for s, n in enumerate(base['objs'])}
res = {}
for st, sr in [(0, 0), (0.7, 0.7), (1, 1), (1.5, 1.5), (2, 2), (3, 3), (1, 0), (0, 1), (2, 0), (0, 2)]:
    o = kitx.score(ep, {n: smooth(X['W'][:, i], st, sr) for i, n in enumerate(X['names'])}, X['meshes'], regs=regs)
    res[f'{st},{sr}'] = {n: (float(o['pos'][:, s].mean()), float(o['ori'][:, s].mean())) for s, n in enumerate(o['objs'])}
json.dump(res, open(f'../out/smooth_{tag}_{ep}.json', 'w'))
