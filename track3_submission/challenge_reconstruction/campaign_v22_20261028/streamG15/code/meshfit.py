"""GT-free mesh refinement from the video: fit a constant body-frame similarity-plus-anisotropic-scale correction of OUR
mesh (3 log-scales along the mesh's own principal axes about its centroid, 3 rotation, 3 translation) that maximises
silhouette IoU against our SAM2 masks over many frames, with our tracked poses held fixed. Occluders (hand masks 2/3 and
other objects' masks) are excluded from the IoU. No reference data is read.
Usage: meshfit.py EPISODE_ROOT EXPORT_TAG OBJECT OUT_DIR [--mode scale|full] [--every 4]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2, trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); ap.add_argument('tag'); ap.add_argument('obj'); ap.add_argument('out', type=Path)
ap.add_argument('--mode', default='full'); ap.add_argument('--every', type=int, default=4); ap.add_argument('--ds', type=int, default=2)
ap.add_argument('--maxscale', type=float, default=0.25)
a = ap.parse_args(); root = a.root; ex = root / 'exports' / a.tag
z = np.load(ex / 'object_poses.npz'); names = [str(n) for n in z['names']]; s = names.index(a.obj)
mj = json.loads((ex / 'meshes.json').read_text()) if (ex / 'meshes.json').exists() else {}
mp = mj.get(a.obj, str(root / 'objects' / a.obj / 'mesh.obj'))
mesh = trimesh.load(mp, force='mesh', process=False); V = np.asarray(mesh.vertices, np.float64); Fc = np.asarray(mesh.faces)
if len(Fc) > 6000:
    ms = mesh.simplify_quadric_decimation(face_count=6000); Vr, Fr = np.asarray(ms.vertices, np.float64), np.asarray(ms.faces)
else: Vr, Fr = V, Fc
c = V.mean(0); w, U = np.linalg.eigh(np.cov((V - c).T))
intr = json.loads((root / 'intrinsics.json').read_text()); ds = a.ds
K = np.array([[intr['fx'] / ds, 0, intr['cx'] / ds], [0, intr['fy'] / ds, intr['cy'] / ds], [0, 0, 1.]]); H, Wd = intr['height'] // ds, intr['width'] // ds
cam_obj = z['camera_from_object'][:, s].astype(np.float64)
obs = z['object_observed'][:, s] if 'object_observed' in z else np.ones(len(cam_obj), bool)
def rd(p):
    m = cv2.imread(str(p), 0)
    return None if m is None else cv2.resize((m > 0).astype(np.uint8), (Wd, H), interpolation=cv2.INTER_NEAREST) > 0
frames = []
for f in range(0, len(cam_obj), a.every):
    if not obs[f]: continue
    m = rd(root / 'objects' / a.obj / 'masks/1' / f'{f:06d}.png')
    if m is None or m.sum() < 1500 / ds ** 2: continue
    # skip masks touching the image border (truncated silhouettes)
    if m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any(): continue
    exc = np.zeros_like(m)
    for lab in '23':
        q = rd(root / 'objects' / a.obj / 'masks' / lab / f'{f:06d}.png')
        if q is not None: exc |= q
    for o in names:
        if o == a.obj: continue
        q = rd(root / 'objects' / o / 'masks/1' / f'{f:06d}.png')
        if q is not None: exc |= q
    exc = (cv2.dilate(exc.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~m
    frames.append((f, m, exc))
if len(frames) < 5: raise SystemExit(f'too few usable frames {len(frames)}')
if len(frames) > 60: frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 60).astype(int)]
def body(p):
    sc = np.exp(np.clip(p[:3], -a.maxscale, a.maxscale)); R = Rot.from_rotvec(p[3:6]).as_matrix(); t = p[6:9]
    A = R @ U @ np.diag(sc) @ U.T
    return A, c + t - A @ c   # v' = A v + b
def render(Vb, T):
    P = Vb @ T[:3, :3].T + T[:3, 3]
    if (P[:, 2] < .05).any(): return None
    uv = (P[:, :2] / P[:, 2:]) @ K[:2, :2].T + K[:2, 2]
    img = np.zeros((H, Wd), np.uint8); cv2.fillPoly(img, list(np.rint(uv[Fr] * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2)
    return img > 0
def cost(p, detail=False):
    A, b = body(p); Vb = Vr @ A.T + b; ious = []
    for f, m, exc in frames:
        r = render(Vb, cam_obj[f])
        if r is None: ious.append(0.); continue
        u = (r | m) & ~exc; i = r & m & ~exc; ious.append(i.sum() / max(1, u.sum()))
    ious = np.array(ious)
    reg = 0.02 * np.sum(np.asarray(p[3:6]) ** 2) / np.deg2rad(5) ** 2 * 0 + 0.0
    return (ious if detail else 1 - ious.mean() + reg)
p0 = np.zeros(9); i0 = cost(p0, True)
mask_fix = {'scale': [0, 1, 2], 'scale_t': [0, 1, 2, 6, 7, 8], 'full': list(range(9)), 't': [6, 7, 8], 'rt': [3, 4, 5, 6, 7, 8]}[a.mode]
def sub(q): p = np.zeros(9); p[mask_fix] = q; return p
best = None
steps = np.array([.05, .05, .05, .03, .03, .03, .005, .005, .005])[mask_fix]
res = minimize(lambda q: cost(sub(q)), np.zeros(len(mask_fix)), method='Powell', options=dict(xtol=1e-3, ftol=1e-4, maxfev=1500, direc=np.diag(steps)))
p = sub(res.x); i1 = cost(p, True); A, b = body(p)
a.out.mkdir(parents=True, exist_ok=True)
Vn = V @ A.T + b
out_mesh = a.out / f'{a.obj}.obj'
trimesh.Trimesh(Vn, Fc, process=False).export(out_mesh)
rep = dict(episode_root=str(root), tag=a.tag, object=a.obj, source_mesh=mp, mode=a.mode, frames=[int(f) for f, _, _ in frames],
           params=p.tolist(), scales=np.exp(np.clip(p[:3], -a.maxscale, a.maxscale)).tolist(), rot_deg=float(np.degrees(np.linalg.norm(p[3:6]))),
           t_cm=(np.asarray(p[6:9]) * 100).tolist(), A=A.tolist(), b=b.tolist(), iou_before=float(i0.mean()), iou_after=float(i1.mean()),
           iou_before_p10=float(np.percentile(i0, 10)), iou_after_p10=float(np.percentile(i1, 10)),
           extent_before_cm=(np.ptp((V - c) @ U, 0) * 100).tolist(), extent_after_cm=(np.ptp((Vn - Vn.mean(0)) @ U, 0) * 100).tolist(),
           ground_truth_used=False)
(a.out / f'{a.obj}_meshfit.json').write_text(json.dumps(rep, indent=1))
print(json.dumps({k: rep[k] for k in ['object', 'mode', 'scales', 'rot_deg', 't_cm', 'iou_before', 'iou_after', 'extent_before_cm', 'extent_after_cm']}), flush=True)
