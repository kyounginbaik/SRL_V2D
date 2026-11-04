#!/usr/bin/env python3
"""ep33 spoon insertion-depth hypothesis test (GT-free). In the scooping windows (stored frames where the bowl is at or in the cup
opening, dilated and ramped), move the spoon along its long axis towards the bowl end by D cm. For each D:
 (1) image fit: silhouette IoU of the visible projected spoon (z-tested against the projected cup) with the SAM2 spoon mask,
     RGB clock, every 2nd frame of the windows;
 (2) a score_vs_compose target (cup and spoon trajectories, meshes from the base target) for kit consistency scoring.
usage: ep33_spoondepth_test.py BUNDLE_DIR BASE_TARGET_EP_DIR OUT_DIR D1,D2,..."""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, gaussian_filter1d

sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA  # noqa: E402
from guard_clip import CAN  # noqa: E402
from retime import shift  # noqa: E402

base, btgt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); Ds = [float(x) for x in sys.argv[4].split(',')]
E = 'episode_000033'; root = CAN / 'evaluation' / E; src = base / E; out.mkdir(parents=True, exist_ok=True)
z = dict(np.load(src / 'object_poses.npz')); names = [str(n) for n in z['names']]; si, ci = names.index('wooden_spoon'), names.index('blue_cup')
F = len(z['camera_from_object']); d = DELTA['evaluation_33']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
mj = json.loads((src / 'meshes.json').read_text())
ms = trimesh.load(src / mj['wooden_spoon']['file'], force='mesh', process=False); mc = trimesh.load(src / mj['blue_cup']['file'], force='mesh', process=False)
VS, FS, VC, FC = np.asarray(ms.vertices), np.asarray(ms.faces), np.asarray(mc.vertices), np.asarray(mc.faces)
# spoon long axis, pointing to the bowl end
c = VS.mean(0); w, U = np.linalg.eigh(np.cov((VS - c).T)); ax = U[:, 2]; s = (VS - c) @ ax; lo, hi = s.min(), s.max(); L = hi - lo
wid = lambda msk: np.linalg.norm((VS[msk] - c) - np.outer(s[msk], ax), axis=1).max()
if wid(s < lo + 0.3 * L) > wid(s > hi - 0.3 * L):
    ax = -ax; s = -s; lo, hi = s.min(), s.max()
bowl_l = VS[s > hi - 0.3 * L].mean(0)
# cup cylinder (body frame)
cc = VC.mean(0); w2, U2 = np.linalg.eigh(np.cov((VC - cc).T)); k = 0 if abs(np.log(w2[1] / w2[0])) > abs(np.log(w2[2] / w2[1])) else 2
cax = U2[:, k]; h = (VC - cc) @ cax; r = np.linalg.norm((VC - cc) - np.outer(h, cax), axis=1); R = np.percentile(r, 95); near = r < 0.5 * R
if near[h < h.min() + 0.15 * (h.max() - h.min())].mean() < near[h > h.max() - 0.15 * (h.max() - h.min())].mean():
    cax = -cax; h = -h
rim = h.max()
Ws, Wc = z['world_from_object'][:, si], z['world_from_object'][:, ci]
bw = Ws[:, :3, 3] + np.einsum('fij,j->fi', Ws[:, :3, :3], bowl_l)
pl = np.einsum('fji,fj->fi', Wc[:, :3, :3], bw - Wc[:, :3, 3]) - cc
hh = pl @ cax; rr = np.linalg.norm(pl - np.outer(hh, cax), axis=1)
win = (rr < R + 0.02) & (hh > rim - 0.06) & (hh < rim + 0.04)
win = binary_dilation(win, iterations=3); wgt = np.clip(gaussian_filter1d(win.astype(float), 2.0) * 1.5, 0, 1)
runs = []; f = 0
while f < F:
    if win[f]:
        g = f
        while g + 1 < F and win[g + 1]:
            g += 1
        runs.append((f, g)); f = g + 1
    else:
        f += 1
print('scooping windows (stored frames):', runs, '| frames', int(win.sum()))
Tt = {n: np.load(btgt / f'{n}_trajectory.npz')['T_world_obj'] for n in ('blue_cup', 'wooden_spoon')}
print('base target vs bundle world_from_object (spoon, max abs diff):', float(np.abs(Tt['wooden_spoon'][:F] - Ws[:len(Tt['wooden_spoon'])]).max()))
proj = lambda X: (X @ K.T)[:, :2] / (X @ K.T)[:, 2:3]
rgbf = sorted({int(round(f + d[f])) for f in range(F) if win[f]} & set(range(0, F)))[::2]
cap = cv2.VideoCapture(str(root / 'rgb.mp4')); ims = {}; t = 0
while t <= max(rgbf):
    ok, im = cap.read()
    if not ok:
        break
    if t in rgbf:
        ims[t] = im.shape[:2]
    t += 1
masks = {f: (cv2.imread(str(root / 'objects/wooden_spoon/masks/1' / f'{f:06d}.png'), cv2.IMREAD_UNCHANGED) > 0) for f in rgbf if (root / 'objects/wooden_spoon/masks/1' / f'{f:06d}.png').exists()}


def fill(uv, faces, H, W, zs=None, zbuf=None):
    sil = np.zeros((H, W), np.uint8)
    for tri in faces:
        cv2.fillConvexPoly(sil, np.round(uv[tri]).astype(np.int32), 1)
    return sil > 0


res = {}
for D in Ds:
    shiftv = wgt[:, None] * (D / 100.0) * ax                         # body-frame offset per stored frame
    camS = z['camera_from_object'][:, si].copy(); camS[:, :3, 3] += np.einsum('fij,fj->fi', camS[:, :3, :3], shiftv)
    worS = z['world_from_object'][:, si].copy(); worS[:, :3, 3] += np.einsum('fij,fj->fi', worS[:, :3, :3], shiftv)
    cS, cC = shift(camS, -d), shift(z['camera_from_object'][:, ci], -d)
    ious = []
    for f in rgbf:
        if f not in masks:
            continue
        H, W = masks[f].shape; TS, TC = cS[f], cC[f]
        XS = VS @ TS[:3, :3].T + TS[:3, 3]; XC = VC @ TC[:3, :3].T + TC[:3, 3]; uvS, uvC = proj(XS), proj(XC)
        spoon = fill(uvS, FS, H, W); cup = fill(uvC, FC, H, W)
        # spoon behind the cup: compare mean depths of spoon pixels covered by cup (coarse z-test per triangle)
        zc = np.full((H, W), np.inf)
        for tri in FC:
            q = np.round(uvC[tri]).astype(np.int32); x0, y0 = np.maximum(q.min(0), 0); x1, y1 = np.minimum(q.max(0), [W - 1, H - 1])
            if x1 < x0 or y1 < y0:
                continue
            sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8); cv2.fillConvexPoly(sub, q - [x0, y0], 1)
            reg = zc[y0:y1 + 1, x0:x1 + 1]; np.minimum(reg, np.where(sub > 0, XC[tri, 2].mean(), np.inf), out=reg)
        zs = np.full((H, W), np.inf)
        for tri in FS:
            q = np.round(uvS[tri]).astype(np.int32); x0, y0 = np.maximum(q.min(0), 0); x1, y1 = np.minimum(q.max(0), [W - 1, H - 1])
            if x1 < x0 or y1 < y0:
                continue
            sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8); cv2.fillConvexPoly(sub, q - [x0, y0], 1)
            reg = zs[y0:y1 + 1, x0:x1 + 1]; np.minimum(reg, np.where(sub > 0, XS[tri, 2].mean(), np.inf), out=reg)
        vis = spoon & (zs < zc - 0.003); m = masks[f]
        ious.append((f, (vis & m).sum() / max(1, (vis | m).sum())))
    res[D] = ious
    tg = out / 'targets' / f'D{D:+.0f}' / E; tg.mkdir(parents=True, exist_ok=True)
    # target: base target spoon trajectory moved by the same body-frame offset (base target and bundle share the spoon body frame)
    Ts = Tt['wooden_spoon'].copy(); Ts[:, :3, 3] += np.einsum('fij,fj->fi', Ts[:, :3, :3], shiftv[:len(Ts)])
    np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=Ts); np.savez_compressed(tg / 'blue_cup_trajectory.npz', T_world_obj=Tt['blue_cup'])
    for mf in ('blue_cup_mesh_metric.obj', 'wooden_spoon_mesh_metric.obj'):
        shutil.copy2(btgt / mf, tg / mf)
print('D (cm) | mean IoU over window frames | per run mean')
fr = np.array([f for f, _ in res[Ds[0]]])
for D in Ds:
    v = np.array([x for _, x in res[D]])
    per = [f'{np.mean([x for f, x in res[D] if a - 2 <= f <= b + 2]):.3f}' for a, b in runs if any(a - 2 <= f <= b + 2 for f, _ in res[D])]
    print(f'{D:+5.1f} | {v.mean():.4f} | {per}')
(out / 'iou.json').write_text(json.dumps({str(D): res[D] for D in Ds}))
