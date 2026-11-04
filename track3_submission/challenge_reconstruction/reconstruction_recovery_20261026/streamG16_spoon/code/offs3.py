"""3D check of the per-episode roll offset (GT-free): the ball detections' stereo camera points vs the model balls under
our pose rolled by theta; (a) plain 3D distance, (b) with a free per-frame 3D translation (absorbs spoon position /
stereo depth bias). Stereo is ~0.9 frame ahead of RGB: early static frames are the clean test.
usage: offs3.py EP"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import ball_model
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis
from scipy.spatial.transform import Rotation as Rot
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); G = RR / 'streamG16_spoon'
EP = int(sys.argv[1]); root = RR / f'canonical/evaluation/episode_{EP:06d}'
d = np.load(RR / 'selected20_final_v10_g12' / f'episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d['names']]
cam = d['camera_from_object'][:, names.index('wooden_spoon')].astype(float)
mesh = trimesh.load(RR / 'selected20_final_v12b_g15' / f'episode_{EP:06d}/meshes/wooden_spoon.obj', force='mesh', process=False)
c, ax = roll_axis(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
model, dets = ball_model(root, 'wooden_spoon')
A = np.load(G / f'out/ana_ep{EP}_l0.05_p20_b2_r1.npz'); inf = A['inf']
th = np.arange(-60, 61, 5.)
fr = [f for f in np.where(inf)[0]]
P = np.full((2, len(fr), len(th)), np.nan)
for i, f in enumerate(fr):
    det = np.array([b['camera_point'] for b in dets[f] if b.get('camera_point') is not None and b.get('depth_mad_m', 1) < 0.006])
    if len(det) < 2: continue
    for j, t in enumerate(th):
        R = Rot.from_rotvec(np.radians(t) * ax).as_matrix(); X = cam[f].copy(); X[:3, :3] = cam[f][:3, :3] @ R; X[:3, 3] = cam[f][:3, :3] @ (c - R @ c) + cam[f][:3, 3]
        m = model @ X[:3, :3].T + X[:3, 3]
        D = np.linalg.norm(det[:, None] - m[None], axis=2); P[0, i, j] = np.median(D.min(1)) * 1000
        s = np.zeros(3)
        for _ in range(5):
            jj = np.linalg.norm(det[:, None] - (m + s)[None], axis=2).argmin(1); s = np.median(det - m[jj], 0)
        P[1, i, j] = np.median(np.linalg.norm(det[:, None] - (m + s)[None], axis=2).min(1)) * 1000
out = {}
for k, lab in enumerate(('plain', 'shift')):
    for nm, msk in (('early', np.array(fr) < 24), ('all', np.ones(len(fr), bool))):
        v = P[k][msk]; v = v[np.isfinite(v).all(1)]
        if len(v) < 3: continue
        cur = np.mean(np.minimum(v, 30), 0)
        rng = np.random.default_rng(0); bs = [th[np.argmin(np.mean(np.minimum(v[rng.integers(0, len(v), len(v))], 30), 0))] for _ in range(300)]
        out[f'{lab}_{nm}'] = dict(n=len(v), curve_min=float(th[np.argmin(cur)]), mm0=round(float(cur[th == 0][0]), 2), mmmin=round(float(cur.min()), 2),
                                  ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))])
print(EP, json.dumps(out)); json.dump(out, open(G / f'out/offs3_ep{EP}.json', 'w'), indent=1)
