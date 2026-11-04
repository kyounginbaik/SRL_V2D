"""Per-frame spoon gray-ball reprojection (px) of an export folder; same model / formula as anchor_audit.ball_check
(model from the episode's spoon_roll_v8 selection, our own detections). No reference data. usage: ballqc.py ROOT FOLDER"""
import sys, json
from pathlib import Path
import numpy as np
root, folder = Path(sys.argv[1]), Path(sys.argv[2])
data = np.load(folder / 'object_poses.npz')
names = [str(n) for n in data['names']]
out = {}
bp = root / 'ball_detections_wooden_spoon.json'
sel = root / 'exports' / 'spoon_roll_v8' / 'selection.json'
if 'wooden_spoon' in names and bp.exists() and sel.exists():
    slot = names.index('wooden_spoon')
    model = np.array(json.loads(sel.read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
    intr = json.loads((root / 'intrinsics.json').read_text())
    k = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    det = json.loads(bp.read_text())['frames']
    cam = data['camera_from_object'][:, slot]
    per = np.full(len(cam), np.nan)
    for f, balls in enumerate(det[:len(cam)]):
        if not balls:
            continue
        pts = model @ cam[f, :3, :3].T + cam[f, :3, 3]
        uv = pts[:, :2] / pts[:, 2:] @ k[:2, :2].T + k[:2, 2]
        per[f] = np.median([np.min(np.linalg.norm(uv - b['uv'], axis=1)) for b in balls])
    out = dict(per_frame=[None if np.isnan(x) else round(float(x), 3) for x in per],
               median=float(np.nanmedian(per)), p90=float(np.nanpercentile(per, 90)), frames=int(np.isfinite(per).sum()))
(folder / 'ballqc.json').write_text(json.dumps(out) + '\n')
print('BALL', folder, {k: v for k, v in out.items() if k != 'per_frame'})
