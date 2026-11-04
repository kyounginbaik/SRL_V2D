"""Z9 GT-free 2D length audit of every non-symmetric object of a pose export (our mesh + our SAM2 masks).

Clean frames only (object mask > 800 px, no hand / other object within 15 px, the whole projected mesh inside the
image, principal axis foreshortening > 0.6). Per frame: mask extent along the projected mesh principal axis vs the
projected mesh extent (ratio) and the two end offsets in mm at the end's depth (+ = mask extends beyond the mesh).
This is a calibration of the measurement: public meshes that the scorer accepts tell what ratio a correct mesh gives.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import trimesh

SYMMETRIC = {'blue_cup', 'beige_cup', 'white_pot_lid'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path); ap.add_argument('poses', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=3)
    a = ap.parse_args(); root = a.root
    data = np.load(a.poses / 'object_poses.npz'); names = [str(n) for n in data['names']]
    meshes = json.loads((a.poses / 'meshes.json').read_text()) if (a.poses / 'meshes.json').exists() else {}
    intr = json.loads((root / 'intrinsics.json').read_text()); W, H = intr['width'], intr['height']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    proj = lambda P, X: (lambda c: c[:, :2] / c[:, 2:] @ K[:2, :2].T + K[:2, 2])(X @ P[:3, :3].T + P[:3, 3])
    rd = lambda p: (cv2.imread(str(p), 0) > 0) if p.exists() else None
    out = dict(root=str(root), poses=str(a.poses), ground_truth_used=False, objects={})
    for slot, name in enumerate(names):
        if name in SYMMETRIC:
            continue
        mv = meshes.get(name, str(root / 'objects' / name / 'mesh.obj'))
        mp = (a.poses / mv['file']) if isinstance(mv, dict) else Path(mv)
        mesh = trimesh.load(mp, force='mesh', process=False)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, 30000, seed=1)[0]); cen = pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((pts - cen).T)); ax = evecs[:, -1]
        s = (pts - cen) @ ax; e0, e1 = cen + s.min() * ax, cen + s.max() * ax
        aspect = float(np.ptp(s) / np.ptp((pts - cen) @ evecs[:, -2]))
        cam = data['camera_from_object'][:, slot]; rows = []
        for t in range(0, len(cam), a.stride):
            mask = rd(root / 'objects' / name / 'masks/1' / f'{t:06d}.png')
            if mask is None or mask.sum() < 800:
                continue
            P = cam[t]; uvm = proj(P, pts)
            if not ((uvm[:, 0] > 5).all() and (uvm[:, 0] < W - 5).all() and (uvm[:, 1] > 5).all() and (uvm[:, 1] < H - 5).all()):
                continue
            near = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            busy = 0
            for p in [root / 'objects' / name / 'masks' / l / f'{t:06d}.png' for l in '23'] + [root / 'objects' / o / 'masks/1' / f'{t:06d}.png' for o in names if o != name]:
                m = rd(p)
                if m is not None:
                    busy += int((m & near).sum())
            if busy >= 50:
                continue
            a0, a1 = proj(P, np.array([e0, e1])); L = np.linalg.norm(a1 - a0)
            z0 = (P[:3, :3] @ e0 + P[:3, 3])[2]; z1 = (P[:3, :3] @ e1 + P[:3, 3])[2]
            fs = L / (K[0, 0] * np.linalg.norm(e1 - e0) / ((z0 + z1) / 2))
            if fs < 0.6:
                continue
            u = (a1 - a0) / L
            vy, vx = np.nonzero(mask); sm = np.stack([vx, vy], 1) @ u; sr = uvm @ u
            lo_m, hi_m = np.percentile(sm, [0.5, 99.5]); lo_r, hi_r = sr.min(), sr.max()
            rows.append(dict(frame=t, ratio=float((hi_m - lo_m) / (hi_r - lo_r)),
                             end0_mm=float((lo_r - lo_m) * z0 / K[0, 0] / fs * 1000), end1_mm=float((hi_m - hi_r) * z1 / K[0, 0] / fs * 1000)))
        st = lambda k: [float(np.median([r[k] for r in rows])), float(np.percentile([r[k] for r in rows], 25)), float(np.percentile([r[k] for r in rows], 75))] if rows else None
        out['objects'][name] = dict(mesh_length_cm=float(np.ptp(s) * 100), aspect=aspect, frames=len(rows), ratio=st('ratio'),
                                    end0_mm=st('end0_mm'), end1_mm=st('end1_mm'), rows=rows)
        print(name, len(rows), st('ratio'), st('end0_mm'), st('end1_mm'), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
