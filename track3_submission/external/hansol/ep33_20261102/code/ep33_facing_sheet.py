#!/usr/bin/env python3
"""ep33 spoon orientation review sheet: for each RGB-clock frame, the RGB crop around the SAM2 spoon mask, and the same crop
with the bundle's spoon outline, its long axis and the concave-side normal (bowl_normal, the same convention as
spoon_annot_roll.py) drawn as an arrow from the bowl centre; the title gives facing = normal . (camera -> bowl), > 0 means
the concave side faces the camera, and the user's mark for that frame if any. Poses are moved to the RGB clock with the
same G15 delta as the roll fit.
usage: ep33_facing_sheet.py BUNDLE_DIR OUT_JPG F0 F1 STEP [ANN_JSON] [--label TEXT]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA  # noqa: E402
from ep32_annot_roll import bowl_centre  # noqa: E402
from guard_clip import CAN  # noqa: E402
from spoon_hand_bowl import bowl_normal  # noqa: E402
from retime import shift  # noqa: E402


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    base, out, f0, f1, st = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    ann = {int(k): v for k, v in json.loads(Path(sys.argv[6]).read_text()).items()} if len(sys.argv) > 6 and not sys.argv[6].startswith('--') else {}
    ep = 33; E = f'episode_{ep:06d}'; root = CAN / 'evaluation' / E; src = base / E
    z = dict(np.load(src / 'object_poses.npz')); s = [str(n) for n in z['names']].index('wooden_spoon')
    F = len(z['camera_from_object']); d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
    I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
    mj = json.loads((src / 'meshes.json').read_text()); mesh = src / mj['wooden_spoon']['file']
    m = trimesh.load(mesh, force='mesh', process=False); pts, _ = trimesh.sample.sample_surface(m, 30000, seed=5); pts = np.asarray(pts, float)
    nb = bowl_normal(str(mesh))[0]; c = m.vertices.mean(0); _, vec = np.linalg.eigh(np.cov((m.vertices - c).T)); ax = vec[:, -1]
    b = bowl_centre(pts, c, ax, nb); cam = shift(z['camera_from_object'][:, s], -d)
    V = np.asarray(m.vertices); Fc = np.asarray(m.faces)
    cap = cv2.VideoCapture(str(root / 'rgb.mp4')); frames = {}
    t = 0
    while t <= f1:
        ok, im = cap.read()
        if not ok:
            break
        if t >= f0:
            frames[t] = im
        t += 1
    proj = lambda P: (P @ K.T)[:, :2] / (P @ K.T)[:, 2:3]
    tiles, CS = [], 300
    for f in range(f0, f1 + 1, st):
        if f not in frames:
            continue
        im = frames[f]; mp = root / 'objects/wooden_spoon/masks/1' / f'{f:06d}.png'
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) > 0 if mp.exists() else np.zeros(im.shape[:2], bool)
        T = cam[f]; P = V @ T[:3, :3].T + T[:3, 3]; uv = proj(P)
        ys, xs = np.nonzero(mask)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        else:
            x0, y0 = uv.min(0).astype(int); x1, y1 = uv.max(0).astype(int)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; half = int(max(x1 - x0, y1 - y0) * 0.65 + 30)
        X0, Y0 = max(0, cx - half), max(0, cy - half); X1, Y1 = min(im.shape[1], cx + half), min(im.shape[0], cy + half)
        a = im[Y0:Y1, X0:X1].copy(); o = a.copy()
        sil = np.zeros(im.shape[:2], np.uint8)
        for tri in Fc:
            cv2.fillConvexPoly(sil, np.round(uv[tri]).astype(np.int32), 1)
        cs_, _ = cv2.findContours(sil[Y0:Y1, X0:X1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs_, -1, (80, 220, 80), 2, cv2.LINE_AA)
        bc = T[:3, :3] @ b + T[:3, 3]; n = T[:3, :3] @ nb
        facing = float(n @ (-bc / np.linalg.norm(bc)))
        q = proj(np.stack([bc, bc + 0.05 * n, T[:3, :3] @ (c - 0.12 * ax) + T[:3, 3], T[:3, :3] @ (c + 0.12 * ax) + T[:3, 3]])) - [X0, Y0]
        q = np.round(q).astype(int)
        cv2.line(o, tuple(q[2]), tuple(q[3]), (80, 220, 80), 1, cv2.LINE_AA)
        cv2.arrowedLine(o, tuple(q[0]), tuple(q[1]), (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)
        a, o = cv2.resize(a, (CS, CS)), cv2.resize(o, (CS, CS))
        tile = np.full((CS + 24, 2 * CS + 4, 3), 20, np.uint8); tile[24:, :CS] = a; tile[24:, CS + 4:] = o
        mk = ann.get(f); mk = (mk if isinstance(mk, str) else f'arrow{mk}') if mk is not None else ''
        col = (120, 255, 120) if facing > 0.3 else ((120, 120, 255) if facing < -0.3 else (230, 230, 230))
        cv2.putText(tile, f'f{f} ({f / 20:.2f}s) facing {facing:+.2f} {mk}', (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tiles.append(tile)
    cols = 4; rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    head = np.full((30, sheet.shape[1], 3), 30, np.uint8)
    cv2.putText(head, f"{arg('--label', base.name)}: RGB | bundle spoon (green), orange arrow = concave-side normal; facing>0 = concave side towards camera",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(out, np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('wrote', out, len(tiles), 'tiles')


if __name__ == '__main__':
    main()
