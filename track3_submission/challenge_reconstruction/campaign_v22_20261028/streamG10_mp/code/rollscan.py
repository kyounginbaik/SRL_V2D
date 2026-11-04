"""GT-free roll profile of the spoon: for every frame, our final_v6 spoon pose rolled about its long (PCA) axis through the
centroid by theta in a grid; evidence per (frame, theta) = 2D gray-ball reprojection (median px, spoon_roll_v8 model) and
visible-mask IoU (Z9/fuse evidence). Compose-independent, so it can say which roll the images prefer on EVERY frame.
usage: rollscan.py EP [STEP_DEG=15] [BUNDLE=selected20_final_v6]"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, ball_model, evidence
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis, C, RR
from scipy.spatial.transform import Rotation as Rot
EP = int(sys.argv[1]); STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 15.
BUN = sys.argv[3] if len(sys.argv) > 3 else 'selected20_final_v6'
root = Path(RR) / f'canonical/evaluation/episode_{EP:06d}'; fol = Path(RR) / BUN / f'episode_{EP:06d}'
d = np.load(fol / 'object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
cam = d['camera_from_object'][:, s]; nf = len(cam)
mesh = trimesh.load(fol / 'meshes/wooden_spoon.obj', force='mesh', process=False)
v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
c, ax = roll_axis(v, f)
pts, fc = trimesh.sample.sample_surface(mesh, 20000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
k, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, 'wooden_spoon')
th = np.arange(0, 360, STEP)
X = []
for t in th:
    R = Rot.from_rotvec(np.radians(t) * ax).as_matrix(); x = np.eye(4); x[:3, :3] = R; x[:3, 3] = c - R @ c; X.append(x)
ball = np.full((nf, len(th)), np.nan); iou = np.full((nf, len(th)), np.nan); nb = np.zeros(nf, int); hand = np.zeros(nf, int)
for t in range(nf):
    b = (dets[t] if (dets is not None and t < len(dets)) else None) or []
    nb[t] = len(b)
    fi = frame_inputs(root, 'wooden_spoon', names, t, 507, 380)
    for j, x in enumerate(X):
        P = cam[t] @ x
        if b: ball[t, j] = ball_px(model, b, P, kf)
        if fi is not None:
            m, ex, dp, hp = fi; hand[t] = hp
            iou[t, j] = evidence(mesh, pts, nrm, P, k, m, ex, dp)[1]['visible_iou']
    if t % 20 == 0: print(t, nb[t], np.round(ball[t], 0).tolist(), np.round(iou[t], 2).tolist(), flush=True)
np.savez(f'{C}/streamG10_mp/out/rollscan_ep{EP}.npz', th=th, ball=ball, iou=iou, nb=nb, hand=hand, axis=ax, centroid=c, bundle=BUN)
print('saved')
