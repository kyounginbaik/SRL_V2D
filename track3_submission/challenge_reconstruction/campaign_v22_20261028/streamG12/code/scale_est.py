"""G12 task C: GT-free mesh scale estimate per object per clip (our data only: our mesh, our pose, our masks, our stereo).
For an unoccluded frame, with our mesh (size s*L) at our pose (depth Zp) and the true object (L, Z):
  silhouette linear ratio a = sqrt(area_rendered / area_observed) ~ s * Z / Zp
  depth ratio            r = median(z_rendered / z_stereo) on shared object pixels ~ Zp / Z
  => s_hat = a * r  (independent of where the tracker put the depth). Biases (mask edge, stereo toward-camera) are
calibrated on public clips by the scorer (validation only).
usage: scale_est.py SPLIT EP POSES_NPZ MESHDIR_OR_JSON OUT_JSON [STRIDE=2]"""
import sys, json, numpy as np, cv2, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k
SPLIT, EP, PNPZ, MS, OUT = sys.argv[1:6]; STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 2
root = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical') / SPLIT / f'episode_{int(EP):06d}'
d = np.load(PNPZ); names = [str(n) for n in d['names']]; W, H = 507, 380
if MS.endswith('.json'): mp = json.load(open(MS))
else: mp = {n: f'{MS}/{n}.obj' for n in names}
k, _ = load_k(root, W, H)
ER = np.ones((5, 5), np.uint8)
out = {}
for s, n in enumerate(names):
    mesh = trimesh.load(mp[n], force='mesh', process=False)
    pts, fc = trimesh.sample.sample_surface(mesh, 120000, seed=1); nrm = np.asarray(mesh.face_normals)[fc]
    cam = d['camera_from_object'][:, s]; rows = []
    for f in range(0, len(cam), STRIDE):
        fi = frame_inputs(root, n, names, f, W, H)
        if fi is None: continue
        m, ex, dp, hp = fi
        P = cam[f]; cl = pts @ P[:3, :3].T + P[:3, 3]; nn = nrm @ P[:3, :3].T
        fac = (np.einsum('ij,ij->i', nn, cl) < 0) & (cl[:, 2] > 0.05); cl = cl[fac]
        if len(cl) < 100: continue
        uv = cl[:, :2] / cl[:, 2:] @ k[:2, :2].T + k[:2, 2]; uv = np.rint(uv).astype(int)
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ins.mean() < 0.999: continue                                     # object must be fully inside the image
        uv, cl = uv[ins], cl[ins]
        z = np.full(H * W, np.inf); np.minimum.at(z, uv[:, 1] * W + uv[:, 0], cl[:, 2]); z = z.reshape(H, W)
        pred = np.isfinite(z)
        reg = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        mm = m > 0
        if mm.sum() < 150: continue
        occ = (reg & ex).sum() / max(1, reg.sum())                           # rendered region hidden by hands / other objects
        mb = (mm[0].any() or mm[-1].any() or mm[:, 0].any() or mm[:, -1].any())
        if occ > 0.03 or mb: continue
        a = np.sqrt(reg.sum() / mm.sum())
        mi = cv2.erode(mm.astype(np.uint8), ER) > 0
        use = pred & mi & ~ex & np.isfinite(dp) & (dp > 0.1) & (dp < 2.0)
        if use.sum() < 40: continue
        r = float(np.median(z[use] / dp[use])); zm = float(np.median(z[use]))
        dz = float(np.median(z[use] - dp[use]))
        iou = float((reg & mm).sum() / max(1, (reg | mm).sum()))
        rows.append([f, a, r, a * r, zm, dz, iou, int(hp)])
    R = np.array(rows) if rows else np.zeros((0, 8))
    good = R[R[:, 6] > 0.5] if len(R) else R
    summ = dict(n_frames=int(len(R)), n_good=int(len(good)))
    if len(good) >= 5:
        summ.update(s_hat=float(np.median(good[:, 3])), s_iqr=[float(np.quantile(good[:, 3], .25)), float(np.quantile(good[:, 3], .75))],
                    a=float(np.median(good[:, 1])), r=float(np.median(good[:, 2])), z=float(np.median(good[:, 4])), dz_mm=float(np.median(good[:, 5]) * 1000),
                    iou=float(np.median(good[:, 6])), static_frac=float(np.mean(good[:, 7] == 0)))
    out[n] = dict(summary=summ, rows=np.round(R, 4).tolist())
    print(SPLIT, EP, n, json.dumps(summ), flush=True)
json.dump(out, open(OUT, 'w'))
