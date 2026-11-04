#!/usr/bin/env python3
"""ep33 cup check (GT-free): (1) extents of the cup meshes in play (bundle mesh used for tracking/physics, the mesh payload in the
submitted parquet), (2) per frame, the bundle cup's projected silhouette vs the SAM2 cup mask: IoU, projected area outside the mask,
mask area not covered. usage: ep33_cupcheck_img.py BUNDLE_DIR PACKAGE_PARQUET FRAMES(a-b:s,...)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
from ep32_robot_vs_video import DELTA  # noqa: E402
from guard_clip import CAN  # noqa: E402
from retime import shift  # noqa: E402

base, pq = Path(sys.argv[1]), Path(sys.argv[2])
frames = sorted({f for r in sys.argv[3].split(',') for f in range(int(r.split(':')[0].split('-')[0]), int(r.split(':')[0].split('-')[1]) + 1, int(r.split(':')[1]))})
E = 'episode_000033'; root = CAN / 'evaluation' / E; src = base / E


def extents(v, lab):
    v = np.asarray(v, float); c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); p = (v - c) @ U
    print(f'{lab}: PCA extents (cm) {np.round(100 * (p.max(0) - p.min(0)), 1).tolist()}, bbox (cm) {np.round(100 * (v.max(0) - v.min(0)), 1).tolist()}, n_vert {len(v)}')


mj = json.loads((src / 'meshes.json').read_text())
for n in ('blue_cup', 'wooden_spoon'):
    extents(trimesh.load(src / mj[n]['file'], force='mesh', process=False).vertices, f'bundle {n} ({mj[n]["file"]})')
df = pd.read_parquet(pq); rows = df[df.row_id.str.startswith('track_3/e000033/w000/f000000/')]
try:
    import public_kit_score as Kp
    dec = getattr(Kp, 'decode', None)
    for _, r in rows.iterrows():
        if dec is not None and r.object_mesh_b64:
            out = dec(r.object_mesh_b64); v = out['vertices'] if isinstance(out, dict) else out[0]
            v = np.asarray(v, float); v = v[np.isfinite(v).all(1)]; v = v[np.abs(v).sum(1) > 0] if len(v) > 32 else v
            extents(v, f'package payload {r.row_id.split("/")[-1]}')
except Exception as ex:
    print('payload decode failed:', repr(ex)[:200])

z = dict(np.load(src / 'object_poses.npz')); ci = [str(n) for n in z['names']].index('blue_cup')
F = len(z['camera_from_object']); d = DELTA['evaluation_33']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
mc = trimesh.load(src / mj['blue_cup']['file'], force='mesh', process=False); VC, FC = np.asarray(mc.vertices), np.asarray(mc.faces)
cam = shift(z['camera_from_object'][:, ci], -d); mdir = next((root / 'objects/blue_cup/masks').iterdir())
proj = lambda X: (X @ K.T)[:, :2] / (X @ K.T)[:, 2:3]
res = []
for f in frames:
    m = cv2.imread(str(mdir / f'{f:06d}.png'), cv2.IMREAD_UNCHANGED)
    if m is None:
        continue
    m = m > 0; H, W = m.shape; T = cam[f]; uv = proj(VC @ T[:3, :3].T + T[:3, 3]); sil = np.zeros((H, W), np.uint8)
    for tri in FC:
        cv2.fillConvexPoly(sil, np.round(uv[tri]).astype(np.int32), 1)
    s = sil > 0; inter = (s & m).sum(); uni = (s | m).sum()
    res.append((f, inter / max(1, uni), (s & ~m).sum() / max(1, s.sum()), (m & ~s).sum() / max(1, m.sum())))
a = np.array(res)
print(f'cup silhouette vs SAM2 cup mask over {len(a)} frames: IoU mean {a[:,1].mean():.3f} (min {a[:,1].min():.3f}); projected-outside-mask {a[:,2].mean():.3f}; mask-not-covered {a[:,3].mean():.3f}')
for f, iou, po, mn in res[::3]:
    print(f'  f{int(f):3d}: IoU {iou:.3f}, projected outside mask {po:.3f}, mask not covered {mn:.3f}')
