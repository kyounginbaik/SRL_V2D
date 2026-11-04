"""G16: GT-free per-frame roll / end-flip scan of the ADOPTED spoon track (selected20_final_v12b_g15) on the RGB clock.
v12b camera_from_object(t) = v10_g12 camera_from_object(t + delta) exactly (G15 retime, verified 0.0 max diff), so the
RGB frame t (ball detections, masks, depth*) is compared with the v10_g12 pose at t (= v12b resampled at t - delta).
(*depth is on the b/c clock, ~0.9 frame ahead; used only as a secondary cue.)
Hypotheses per frame: pose @ X(flip, theta) with X = roll theta about the long PCA axis through the surface centroid,
optionally preceded by a 180 deg end-to-end flip about the 2nd PCA axis (tip <-> handle).
Evidence: 2D gray-ball reprojection px (median over detections of nearest projected model ball; shared spoon ball model)
for theta step BSTEP; silhouette visible IoU + stereo depth residual for theta step ISTEP.
usage: g16scan.py EP F0 F1 [BSTEP=5] [ISTEP=30]"""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, ball_model, evidence
from scipy.spatial.transform import Rotation as Rot
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
G = RR / 'streamG16_spoon'
EP, F0, F1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
BSTEP = float(sys.argv[4]) if len(sys.argv) > 4 else 5.; ISTEP = float(sys.argv[5]) if len(sys.argv) > 5 else 30.
root = RR / f'canonical/evaluation/episode_{EP:06d}'
src = RR / 'selected20_final_v10_g12' / f'episode_{EP:06d}'
d = np.load(src / 'object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
cam = d['camera_from_object'][:, s].astype(float); nf = len(cam); F1 = min(F1, nf)
mesh = trimesh.load(RR / 'selected20_final_v12b_g15' / f'episode_{EP:06d}/meshes/wooden_spoon.obj', force='mesh', process=False)
v = np.asarray(mesh.vertices, float)
x = v - v.mean(0); w, U = np.linalg.eigh(x.T @ x); ax1, ax2 = U[:, -1], U[:, -2]
import sys as _s; _s.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import roll_axis
c, axr = roll_axis(v, np.asarray(mesh.faces))     # same axis/centroid as G10 rollscan (sign may differ)
ax1 = axr
pts, fc = trimesh.sample.sample_surface(mesh, 20000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
k, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
model, dets = ball_model(root, 'wooden_spoon')


def X(flip, th):
    R = Rot.from_rotvec(np.radians(th) * ax1).as_matrix()
    if flip: R = R @ Rot.from_rotvec(np.pi * ax2).as_matrix()
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c - R @ c; return M


thb = np.arange(0, 360, BSTEP); thi = np.arange(0, 360, ISTEP)
XB = [[X(f, t) for t in thb] for f in (0, 1)]; XI = [[X(f, t) for t in thi] for f in (0, 1)]
n = F1 - F0
ball = np.full((n, 2, len(thb)), np.nan); iou = np.full((n, 2, len(thi)), np.nan); dz = np.full((n, 2, len(thi)), np.nan)
nb = np.zeros(n, int); hand = np.zeros(n, int); mpx = np.zeros(n, int)
for i, t in enumerate(range(F0, F1)):
    b = (dets[t] if (dets is not None and t < len(dets)) else None) or []
    nb[i] = len(b)
    if b:
        for f in (0, 1):
            for j, M in enumerate(XB[f]): ball[i, f, j] = ball_px(model, b, cam[t] @ M, kf)
    fi = frame_inputs(root, 'wooden_spoon', names, t, 507, 380)
    if fi is not None:
        m, ex, dp, hp = fi; hand[i] = hp; mpx[i] = int(m.sum())
        for f in (0, 1):
            for j, M in enumerate(XI[f]):
                _, e = evidence(mesh, pts, nrm, cam[t] @ M, k, m, ex, dp)
                iou[i, f, j] = e['visible_iou']; dz[i, f, j] = e['median_z_mm'] if e['median_z_mm'] is not None else np.nan
    if i % 25 == 0: print(t, nb[i], np.round(ball[i, 0, ::6], 0).tolist(), np.round(iou[i, 0], 2).tolist(), flush=True)
np.savez(G / f'out/scan_ep{EP}_{F0:04d}.npz', frames=np.arange(F0, F1), thb=thb, thi=thi, ball=ball, iou=iou, dz=dz, nb=nb, hand=hand,
         mask_px=mpx, axis=ax1, axis2=ax2, centroid=c, source=str(src))
print('saved')
