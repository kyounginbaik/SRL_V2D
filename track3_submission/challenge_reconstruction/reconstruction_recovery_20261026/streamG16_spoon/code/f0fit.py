"""G16 frame-0 spoon pose check (GT-free): over the early static window (frames f < W where the adopted spoon moves
< 1 cm / 3 deg in world), fit ONE body-frame correction (rotation about the surface centroid, optional translation)
to the 2D gray-ball detections (robust nearest-ball reprojection, soft-L1) and, separately, to the stereo 3D ball points.
Reports the correction as roll about the long axis + tilt of the long axis, and the px / mm before -> after.
Kit relevance: where the spoon is the scene anchor and slot 0 is another object (eps 32, 33), a frame-0 spoon rotation
error theta moves every other object by ~ d * theta (d = 21 cm) and the spoon itself the same about the cup.
usage: f0fit.py EP [W=24] [force: ignore the static-window cut; the body correction is assumed constant]"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import load_k, ball_model
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); G = RR / 'streamG16_spoon'
EP = int(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 24
root = RR / f'canonical/evaluation/episode_{EP:06d}'
d = np.load(RR / 'selected20_final_v10_g12' / f'episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
cam = d['camera_from_object'][:, s].astype(float); wo = d['world_from_object'][:, s].astype(float)
mesh = trimesh.load(RR / 'selected20_final_v12b_g15' / f'episode_{EP:06d}/meshes/wooden_spoon.obj', force='mesh', process=False)
c, ax = roll_axis(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
_, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, 'wooden_spoon')
# static window
dp = np.linalg.norm(wo[:, :3, 3] - wo[0, :3, 3], axis=1) * 100
dr = np.degrees(np.linalg.norm((Rot.from_matrix(wo[:, :3, :3]) * Rot.from_matrix(wo[0, :3, :3]).inv()).as_rotvec(), axis=1))
Wst = int(np.argmax((dp > 1) | (dr > 3))) if ((dp > 1) | (dr > 3)).any() else len(wo)
W = W if len(sys.argv) > 3 and sys.argv[3] == "force" else min(W, Wst)
fr = [f for f in range(W) if dets[f] and len(dets[f]) >= 2]
D2 = [np.array([b['uv'] for b in dets[f]]) for f in fr]
D3 = [np.array([b['camera_point'] for b in dets[f] if b.get('camera_point') is not None and b.get('depth_mad_m', 1) < 0.006]) for f in fr]


def corr(p):
    R = Rot.from_rotvec(np.radians(p[:3])).as_matrix(); X = np.eye(4); X[:3, :3] = R; X[:3, 3] = c - R @ c + (p[3:6] / 100 if len(p) > 3 else 0); return X


def res2(p, cap=30.):
    X = corr(p); r = []
    for f, det in zip(fr, D2):
        P = cam[f] @ X; q = model @ P[:3, :3].T + P[:3, 3]; uv = q[:, :2] / q[:, 2:] @ kf[:2, :2].T + kf[:2, 2]
        e = np.linalg.norm(det[:, None] - uv[None], axis=2); j = e.argmin(1); r.append(np.clip((det - uv[j]).ravel(), -cap, cap))
    return np.concatenate(r)


def res3(p):
    X = corr(p); r = []
    for f, det in zip(fr, D3):
        if len(det) < 2: continue
        P = cam[f] @ X; q = model @ P[:3, :3].T + P[:3, 3]
        e = np.linalg.norm(det[:, None] - q[None], axis=2); j = e.argmin(1); r.append((det - q[j]).ravel() * 1000)
    return np.concatenate(r)


def decomp(p):
    R = Rot.from_rotvec(np.radians(p[:3])); a2 = R.apply(ax)
    tilt = float(np.degrees(np.arccos(np.clip(abs(a2 @ ax), -1, 1))))
    # roll = swing-twist twist about ax
    q = R.as_quat(); tw = np.array([*(ax * (q[:3] @ ax)), q[3]]); tw /= np.linalg.norm(tw); roll = float(np.degrees(Rot.from_quat(tw).as_rotvec() @ ax))
    return dict(roll=round(roll, 1), tilt=round(tilt, 1), total=round(float(np.degrees(R.magnitude())), 1))


def med(fn, p):
    r = fn(p); k = 2 if fn is res2 else 3; return round(float(np.median(np.linalg.norm(r.reshape(-1, k), axis=1))), 2)


out = dict(ep=EP, static_window=Wst, frames=len(fr), window=W)
for lab, fn in (('2d', res2), ('3d', res3)):
    best = {}
    for r0 in range(-60, 61, 10):
        for npar in (3, 6):
            p0 = np.zeros(npar); p0[:3] = ax * r0
            sol = least_squares(fn, p0, loss='soft_l1', f_scale=3.0)
            if npar not in best or sol.cost < best[npar].cost: best[npar] = sol
    for npar, sol in best.items():
        key = f'{lab}_{"rot" if npar == 3 else "rot+trans"}'
        out[key] = dict(**decomp(sol.x), trans_cm=np.round(sol.x[3:], 2).tolist() if npar == 6 else None,
                        err_before=med(fn, np.zeros(npar)), err_after=med(fn, sol.x))
        # roll-only refit for reference
    print(EP, lab, {k: v for k, v in out.items() if k.startswith(lab)}, flush=True)
print(EP, 'static window', Wst, 'frames used', len(fr))
json.dump(out, open(G / f'out/f0fit_ep{EP}_w{W}.json', 'w'), indent=1)
