"""Z9: per-clip constant body-frame silhouette refit (GT-free; our mesh, our SAM2 masks, no stereo depth).

For one object: clean frames (mask >= 800 px at full res, no hand / other object mask within 15 px, the projected
mesh fully inside the image), up to 40 evenly spaced. Find a constant body-frame correction D (rotation about the
mesh centroid, |r| <= 5 deg, translation |t| <= 2 cm) maximizing the mean half-resolution silhouette IoU
(fuse.evidence) over those frames: world' = world @ D on every frame. Accepted only if
  - >= 10 clean frames, mean IoU gain >= 0.02 and the gain is positive on >= 75 % of the frames,
  - for an object with a gray-ball model (spoon) the median ball reprojection does not get worse by > 1 px.
Otherwise the export is copied unchanged.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_model, ball_px  # noqa: E402
from fuse import evidence  # noqa: E402

SYMMETRIC = {'blue_cup', 'beige_cup', 'white_pot_lid'}


def body_delta(x, centroid):
    D = np.eye(4); Rm = Rotation.from_rotvec(np.radians(x[:3])).as_matrix()
    D[:3, :3] = Rm; D[:3, 3] = centroid - Rm @ centroid + x[3:] / 100.
    return D


def clean_frames(root, name, names, cam, pts, K, W, H, n_max=40):
    rd = lambda p: (cv2.imread(str(p), 0) > 0) if p.exists() else None
    ok = []
    for t in range(0, len(cam), 2):
        mask = rd(root / 'objects' / name / 'masks/1' / f'{t:06d}.png')
        if mask is None or mask.sum() < 800:
            continue
        P = cam[t]; c = pts @ P[:3, :3].T + P[:3, 3]
        if (c[:, 2] < .05).any():
            continue
        uv = c[:, :2] / c[:, 2:] @ K[:2, :2].T + K[:2, 2]
        if not ((uv[:, 0] > 5).all() and (uv[:, 0] < W - 5).all() and (uv[:, 1] > 5).all() and (uv[:, 1] < H - 5).all()):
            continue
        near = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        busy = 0
        for p in [root / 'objects' / name / 'masks' / l / f'{t:06d}.png' for l in '23'] + [root / 'objects' / o / 'masks/1' / f'{t:06d}.png' for o in names if o != name]:
            m = rd(p)
            if m is not None:
                busy += int((m & near).sum())
        if busy < 50:
            ok.append(t)
    if len(ok) > n_max:
        ok = [ok[i] for i in np.linspace(0, len(ok) - 1, n_max).round().astype(int)]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path); ap.add_argument('poses', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--objects', default='nonsym', help="'nonsym' or comma list")
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args(); root = a.root
    data = dict(np.load(a.poses / 'object_poses.npz')); names = [str(n) for n in data['names']]
    meshes = json.loads((a.poses / 'meshes.json').read_text()) if (a.poses / 'meshes.json').exists() else {}
    intr = json.loads((root / 'intrinsics.json').read_text()); W, H = intr['width'], intr['height']
    Kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    k, _ = load_k(root, 507, 380)
    targets = [n for n in names if n not in SYMMETRIC] if a.objects == 'nonsym' else [n for n in a.objects.split(',') if n in names]
    world = data['world_from_object'].copy(); rep = dict(root=str(root), poses=str(a.poses), ground_truth_used=False, stereo_used=False, objects={})
    cv2.setNumThreads(2)
    for name in targets:
        slot = names.index(name)
        mv = meshes.get(name, str(root / 'objects' / name / 'mesh.obj'))
        mesh = trimesh.load((a.poses / mv['file']) if isinstance(mv, dict) else Path(mv), force='mesh', process=False)
        pts, fc = trimesh.sample.sample_surface(mesh, 12000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
        centroid = np.asarray(mesh.centroid)
        cam = data['camera_from_object'][:, slot]
        frames = clean_frames(root, name, names, cam, np.asarray(pts)[::10], Kf, W, H)
        r = dict(clean_frames=frames, accepted=False)
        if len(frames) < 10:
            r['reason'] = 'fewer than 10 clean frames'; rep['objects'][name] = r; continue
        inputs = {t: frame_inputs(root, name, names, t, 507, 380) for t in frames}
        model, dets = ball_model(root, name)
        def ious(x):
            D = body_delta(x, centroid)
            return np.array([evidence(mesh, pts, nrm, cam[t] @ D, k, inputs[t][0], inputs[t][1], inputs[t][2])[1]['visible_iou'] for t in frames])
        def balls(x):
            if model is None:
                return np.nan
            D = body_delta(x, centroid)
            v = [ball_px(model, dets[t], cam[t] @ D, Kf) for t in range(len(cam)) if t < len(dets) and dets[t]][::3]
            return float(np.nanmedian(v)) if v else np.nan
        f = lambda x: -ious(np.clip(x, [-5] * 3 + [-2] * 3, [5] * 3 + [2] * 3)).mean()
        base = ious(np.zeros(6))
        res = minimize(f, np.zeros(6), method='Powell', options=dict(xtol=0.05, ftol=1e-4, maxfev=600))
        x = np.clip(res.x, [-5] * 3 + [-2] * 3, [5] * 3 + [2] * 3); new = ious(x)
        gain = new - base; b0, b1 = balls(np.zeros(6)), balls(x)
        r.update(x_rot_deg=x[:3].round(3).tolist(), x_t_cm=x[3:].round(3).tolist(), iou_before=float(base.mean()), iou_after=float(new.mean()),
                 gain_positive_fraction=float(np.mean(gain > 0)), ball_px_before=b0, ball_px_after=b1, nfev=int(res.nfev))
        ok = new.mean() - base.mean() >= 0.02 and np.mean(gain > 0) >= 0.75 and (np.isnan(b0) or b1 <= b0 + 1.)
        r['accepted'] = bool(ok)
        if ok:
            D = body_delta(x, centroid); world[:, slot] = world[:, slot] @ D
            r['centroid_shift_mm'] = float(np.linalg.norm(D[:3, :3] @ centroid + D[:3, 3] - centroid) * 1000)
            r['rot_deg'] = float(np.linalg.norm(x[:3]))
        rep['objects'][name] = r
        print(name, {kk: v for kk, v in r.items() if kk != 'clean_frames'}, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for p in a.poses.iterdir():
        if p.is_file() and p.name not in ('object_poses.npz', 'public_score.json', 'qc.json'):
            shutil.copy2(p, a.out / p.name)
        elif p.is_dir() and not (a.out / p.name).exists():
            shutil.copytree(p, a.out / p.name)
    data['world_from_object'] = world
    data['camera_from_object'] = np.linalg.inv(data['world_from_rgb'])[:, None] @ world
    np.savez_compressed(a.out / 'object_poses.npz', **data)
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
