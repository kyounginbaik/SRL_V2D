"""GT-free check: which pose time does the stereo depth of frame t agree with? For shifts d, render our mesh at the
pose resampled at t+d and compute the median |z_render - z_stereo| over rendered pixels (hands/other objects excluded,
object mask NOT required), on frames where the object moves fast (our own track speed). Usage EP_ROOT TAG"""
import sys, json, numpy as np, cv2, trimesh
from pathlib import Path
from retime import shift
root = Path(sys.argv[1]); tag = sys.argv[2]; ex = root / 'exports' / tag
z = np.load(ex / 'object_poses.npz'); names = [str(n) for n in z['names']]
mj = json.loads((ex / 'meshes.json').read_text()) if (ex / 'meshes.json').exists() else {}
intr = json.loads((root / 'intrinsics.json').read_text()); ds = 2
K = np.array([[intr['fx'] / ds, 0, intr['cx'] / ds], [0, intr['fy'] / ds, intr['cy'] / ds], [0, 0, 1.]]); H, W = intr['height'] // ds, intr['width'] // ds
def rd(p):
    m = cv2.imread(str(p), 0); return None if m is None else cv2.resize((m > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
for s, n in enumerate(names):
    mp = mj.get(n, str(root / 'objects' / n / 'mesh.obj')); mp = mp['file'] if isinstance(mp, dict) else mp
    mesh = trimesh.load(mp, force='mesh', process=False)
    if len(mesh.faces) > 5000: mesh = mesh.simplify_quadric_decimation(face_count=5000)
    V, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    C = z['camera_from_object'][:, s].astype(float); Wd = z['world_from_object'][:, s]
    sp = np.r_[0, np.linalg.norm(np.diff(Wd[:, :3, 3], axis=0), axis=1)] * 20
    frames = [f for f in range(2, len(C) - 3) if sp[f] > 0.15]
    if len(frames) > 80: frames = frames[::len(frames) // 80 + 1]
    res = {}
    for d in [-1.8, -0.9, -0.45, 0, 0.45, 0.9, 1.35, 1.8]:
        Cd = shift(C, d); r = []
        for f in frames:
            enc = cv2.imread(str(root / 'depth' / f'{f:06d}.png'), -1)
            dep = cv2.resize(65535 / np.maximum(enc.astype(float), 1) - 1, (W, H), interpolation=cv2.INTER_NEAREST); dep[cv2.resize(((enc == 0) | (enc >= 65535)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0] = np.nan
            exc = np.zeros((H, W), bool)
            for lab in '23':
                q = rd(root / 'objects' / n / 'masks' / lab / f'{f:06d}.png')
                if q is not None: exc |= q
            for o in names:
                if o != n:
                    q = rd(root / 'objects' / o / 'masks/1' / f'{f:06d}.png')
                    if q is not None: exc |= q
            P = V @ Cd[f, :3, :3].T + Cd[f, :3, 3]
            if (P[:, 2] < .05).any(): continue
            uv = P[:, :2] / P[:, 2:] @ K[:2, :2].T + K[:2, 2]
            zb = np.full((H, W), np.inf)
            # z-buffer by splatting dense surface samples
            pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(P, Fc, process=False), 20000, seed=1)
            u = np.rint(pts[:, :2] / pts[:, 2:] @ K[:2, :2].T + K[:2, 2]).astype(int); ok = (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)
            np.minimum.at(zb, (u[ok, 1], u[ok, 0]), pts[ok, 2])
            use = np.isfinite(zb) & np.isfinite(dep) & ~exc
            if use.sum() < 30: continue
            r.append(np.median(np.abs(zb[use] - dep[use])) * 1000)
        res[d] = float(np.mean(r)) if r else None
    print(root.name[-2:], n, len(frames), ' '.join(f'{d:+.2f}:{v:.2f}' for d, v in res.items() if v is not None), flush=True)
