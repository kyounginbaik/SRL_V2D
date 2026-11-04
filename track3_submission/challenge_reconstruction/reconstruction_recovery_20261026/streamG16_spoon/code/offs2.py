"""Robustness of the per-episode constant roll offset (GT-free): ball fit per frame over roll theta with
(a) the plain ball px, (b) a free per-frame 2D image shift (absorbs spoon translation / depth bias, which a roll could
otherwise mimic), (c) the two neck balls only (tip ball dropped). Frames: informative frames of the analysis.
usage: offs2.py EP [FMAX]"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import load_k, ball_model
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis
from scipy.spatial.transform import Rotation as Rot
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); G = RR / 'streamG16_spoon'
EP = int(sys.argv[1]); FMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10**6
root = RR / f'canonical/evaluation/episode_{EP:06d}'
d = np.load(RR / 'selected20_final_v10_g12' / f'episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d['names']]
cam = d['camera_from_object'][:, names.index('wooden_spoon')].astype(float)
mesh = trimesh.load(RR / 'selected20_final_v12b_g15' / f'episode_{EP:06d}/meshes/wooden_spoon.obj', force='mesh', process=False)
c, ax = roll_axis(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
_, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, 'wooden_spoon')
A = np.load(G / f'out/ana_ep{EP}_l0.05_p20_b2_r1.npz'); inf = A['inf']
th = np.arange(-60, 61, 5.)
def proj(P, M):
    p = M @ P[:3, :3].T + P[:3, 3]; return p[:, :2] / p[:, 2:] @ kf[:2, :2].T + kf[:2, 2]
def score(uv, det, shift):
    s = np.zeros(2)
    for _ in range(5 if shift else 1):
        D = np.linalg.norm((uv + s)[None] - det[:, None], axis=2); j = D.argmin(1)
        if shift: s = np.median(det - uv[j], 0)
    D = np.linalg.norm((uv + s)[None] - det[:, None], axis=2); return float(np.median(D.min(1)))
fr = [f for f in np.where(inf)[0] if f < FMAX]
res = {k: np.full((len(fr), len(th)), np.nan) for k in ('plain', 'shift', 'neck')}
for i, f in enumerate(fr):
    det = np.array([b['uv'] for b in dets[f]])
    for j, t in enumerate(th):
        R = Rot.from_rotvec(np.radians(t) * ax).as_matrix(); X = np.eye(4); X[:3, :3] = R; X[:3, 3] = c - R @ c
        uv = proj(cam[f] @ X, model)
        res['plain'][i, j] = score(uv, det, False); res['shift'][i, j] = score(uv, det, True)
        res['neck'][i, j] = score(uv[:2], det, False)
out = {}
for k, v in res.items():
    for lab, m in (('early', np.array(fr) < 24), ('all', np.ones(len(fr), bool))):
        if m.sum() == 0: continue
        cur = np.mean(np.minimum(v[m], 40), 0); best = th[np.argmin(np.minimum(v[m], 40), 1)]
        out[f'{k}_{lab}'] = dict(n=int(m.sum()), curve_min=float(th[np.argmin(cur)]), px0=round(float(cur[th == 0][0]), 2), pxmin=round(float(cur.min()), 2),
                                 median_best=float(np.median(best)))
print(EP, json.dumps(out)); json.dump(out, open(G / f'out/offs2_ep{EP}.json', 'w'), indent=1)
