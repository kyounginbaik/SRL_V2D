#!/usr/bin/env python3
"""ep33: is the reconstructed spoon bowl as deep in the cup as the RGB shows? (GT-free; RGB, SAM2 masks and stereo only)
For RGB-clock frames in the scooping windows: project the bundle's spoon (bowl part = wide end of the long PCA axis, last 30 %) and cup
into the RGB, z-test the spoon against the cup surface (so a bowl inside the cup is hidden by the cup wall), and compare with the SAM2
spoon mask: visible-bowl pixels inside / outside the spoon mask, and the image offset between the mask and the visible projected
spoon. Writes a sheet (RGB crop | crop with the projected spoon outline green, visible bowl orange, cup outline blue) and a table.
usage: ep33_cupdepth_check.py BUNDLE_DIR OUT_DIR FRAMES(a-b:s,...)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA  # noqa: E402
from guard_clip import CAN  # noqa: E402
from retime import shift  # noqa: E402

base, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
frames = sorted({f for r in sys.argv[3].split(',') for f in range(int(r.split(':')[0].split('-')[0]), int(r.split(':')[0].split('-')[1]) + 1, int(r.split(':')[1]))})
ep = 33; E = f'episode_{ep:06d}'; root = CAN / 'evaluation' / E; src = base / E
z = dict(np.load(src / 'object_poses.npz')); names = [str(n) for n in z['names']]; si, ci = names.index('wooden_spoon'), names.index('blue_cup')
F = len(z['camera_from_object']); d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
mj = json.loads((src / 'meshes.json').read_text())
ms = trimesh.load(src / mj['wooden_spoon']['file'], force='mesh', process=False); mc = trimesh.load(src / mj['blue_cup']['file'], force='mesh', process=False)
camS = shift(z['camera_from_object'][:, si], -d); camC = shift(z['camera_from_object'][:, ci], -d)
VS, FS, VC, FC = np.asarray(ms.vertices), np.asarray(ms.faces), np.asarray(mc.vertices), np.asarray(mc.faces)
c = VS.mean(0); w, U = np.linalg.eigh(np.cov((VS - c).T)); ax = U[:, 2]; s = (VS - c) @ ax; lo, hi = s.min(), s.max(); L = hi - lo
wid = lambda msk: np.linalg.norm((VS[msk] - c) - np.outer(s[msk], ax), axis=1).max()
bowl = (s > hi - 0.3 * L) if wid(s > hi - 0.3 * L) > wid(s < lo + 0.3 * L) else (s < lo + 0.3 * L)
P, _ = trimesh.sample.sample_surface(ms, 20000, seed=3); P = np.asarray(P); sp = (P - c) @ ax
Pbowl = (sp > hi - 0.3 * L) if bowl[np.argmax(s)] else (sp < lo + 0.3 * L)
proj = lambda X: (X @ K.T)[:, :2] / (X @ K.T)[:, 2:3]
cap = cv2.VideoCapture(str(root / 'rgb.mp4')); ims = {}; t = 0
while t <= frames[-1]:
    ok, im = cap.read()
    if not ok:
        break
    if t in frames:
        ims[t] = im
    t += 1
cup_mask_dir = next((root / 'objects/blue_cup/masks').iterdir())
rows, tiles, CS = [], [], 300
for f in frames:
    im = ims[f]; H, W = im.shape[:2]
    TS, TC = camS[f], camC[f]; XS = VS @ TS[:3, :3].T + TS[:3, 3]; XC = VC @ TC[:3, :3].T + TC[:3, 3]
    uvS, uvC = proj(XS), proj(XC)
    zc = np.full((H, W), np.inf)                       # cup depth buffer (triangle mean depth)
    for tri in FC:
        q = np.round(uvC[tri]).astype(np.int32); x0, y0 = q.min(0); x1, y1 = q.max(0)
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1); sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillConvexPoly(sub, q - [x0, y0], 1); zz = XC[tri, 2].mean(); reg = zc[y0:y1 + 1, x0:x1 + 1]; np.minimum(reg, np.where(sub > 0, zz, np.inf), out=reg)
    XP = P @ TS[:3, :3].T + TS[:3, 3]; uvP = np.round(proj(XP)).astype(int); okp = (uvP[:, 0] >= 0) & (uvP[:, 0] < W) & (uvP[:, 1] >= 0) & (uvP[:, 1] < H)
    vis = okp.copy(); vis[okp] = XP[okp, 2] < zc[uvP[okp, 1], uvP[okp, 0]] - 0.003
    sm = cv2.imread(str(root / 'objects/wooden_spoon/masks/1' / f'{f:06d}.png'), cv2.IMREAD_UNCHANGED); sm = (sm > 0) if sm is not None else np.zeros((H, W), bool)
    cm = cv2.imread(str(cup_mask_dir / f'{f:06d}.png'), cv2.IMREAD_UNCHANGED); cm = (cm > 0) if cm is not None else np.zeros((H, W), bool)
    vb = vis & Pbowl; inm = sm[uvP[vb, 1], uvP[vb, 0]] if vb.any() else np.array([])
    bowl_vis = vb.sum() / max(1, (Pbowl & okp).sum()); bowl_in_mask = float(inm.mean()) if len(inm) else float('nan')
    # mask pixels near the bowl end that the projected spoon does not explain (a real bowl lower/elsewhere)
    ys, xs = np.nonzero(sm); msk_c = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])
    vc_ = uvP[vis & okp].mean(0) if (vis & okp).any() else np.array([np.nan, np.nan])
    rows.append(dict(frame=f, t=f / 20, bowl_visible=round(float(bowl_vis), 2), visible_bowl_in_spoon_mask=round(bowl_in_mask, 2),
                     mask_minus_proj_px=[round(float(v), 1) for v in (msk_c - vc_)]))
    ys2, xs2 = np.nonzero(sm | cm); cx, cy = (int(np.median(xs2)), int(np.median(ys2))) if len(xs2) else (W // 2, H // 2)
    half = 170; X0, Y0, X1, Y1 = max(0, cx - half), max(0, cy - half), min(W, cx + half), min(H, cy + half)
    a = im[Y0:Y1, X0:X1].copy(); o = a.copy()
    for verts, faces, col in ((uvC, FC, (255, 120, 40)), (uvS, FS, (80, 220, 80))):
        sil = np.zeros((H, W), np.uint8)
        for tri in faces:
            cv2.fillConvexPoly(sil, np.round(verts[tri]).astype(np.int32), 1)
        cs_, _ = cv2.findContours(sil[Y0:Y1, X0:X1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs_, -1, col, 2, cv2.LINE_AA)
    for (u, v) in uvP[vb][::6]:
        if X0 <= u < X1 and Y0 <= v < Y1:
            o[v - Y0, u - X0] = (0, 160, 255)
    a, o = cv2.resize(a, (CS, CS)), cv2.resize(o, (CS, CS))
    tile = np.full((CS + 24, 2 * CS + 4, 3), 20, np.uint8); tile[24:, :CS] = a; tile[24:, CS + 4:] = o
    cv2.putText(tile, f'f{f} ({f/20:.2f}s) bowl visible {bowl_vis:.2f}, in mask {bowl_in_mask:.2f}', (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    tiles.append(tile)
C = 4; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % C)
g = np.vstack([np.hstack(tiles[i:i + C]) for i in range(0, len(tiles), C)])
head = np.full((30, g.shape[1], 3), 20, np.uint8)
cv2.putText(head, f'{base.name}: RGB | projected spoon (green), visible bowl points (orange, after z-test vs cup), cup outline (blue)', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
cv2.imwrite(str(out / 'cupdepth_sheet.jpg'), np.vstack([head, g]), [cv2.IMWRITE_JPEG_QUALITY, 85])
(out / 'cupdepth.json').write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r)
