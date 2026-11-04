"""G17 rule W (GT-free): window-level adoption of an independent tracker where the image clearly prefers it.
On the RGB clock (pre-retime poses), for each non-spoon object and tracker h in {raw, multimap, pointmap_cotracker}:
  a window = run of >= MINLEN frames where h differs from ours by > 2 cm or > 15 deg (cups: axis angle);
  adopt h on the window iff mean IoU(h) >= mean IoU(ours) + DIOU on the window's frames (every frame scored) AND median
  stereo |depth residual| of h is not worse than ours by more than 0.3 cm; best tracker wins; 3-frame SE(3) blend at both ends.
Usage: winswap.py SPLIT EP SRC_NPZ MESHES_JSON DST_NPZ [--diou 0.1] [--minlen 3]"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import *
ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('ep', type=int); ap.add_argument('src', type=Path); ap.add_argument('meshes', type=Path); ap.add_argument('dst', type=Path)
ap.add_argument('--diou', type=float, default=0.1); ap.add_argument('--minlen', type=int, default=3); ap.add_argument('--blend', type=int, default=3)
a = ap.parse_args()
if a.dst.exists(): raise SystemExit('exists')
root = eproot(a.split, a.ep); E = Episode(root); z, names = load_poses(a.src); F = len(z['world_from_rgb'])
mj = json.loads(a.meshes.read_text()); mp = lambda n: (a.meshes.parent / mj[n]['file']) if isinstance(mj[n], dict) else mj[n]
M = {n: Mesh(mp(n)) for n in names}; AX = {'blue_cup', 'beige_cup'}
C = z['camera_from_object'].astype(float).copy(); cam = z['world_from_rgb'].astype(float)
hyps = {}
for t in ['raw', 'multimap', 'pointmap_cotracker']:
    p = root / 'exports' / t / 'object_poses.npz'
    if p.exists():
        h = np.load(p); hn = [str(n) for n in h['names']]
        hyps[t] = {n: h['camera_from_object'][:F, hn.index(n)].astype(float) for n in names if n in hn}
def symax(n):
    w, U = np.linalg.eigh(np.cov((M[n].V - M[n].c).T)); g = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) for i in range(3)]; return U[:, int(np.argmax(g))]
def dis(T1, T2, n):
    c = M[n].c; dp = np.linalg.norm((T1[:3, :3] @ c + T1[:3, 3]) - (T2[:3, :3] @ c + T2[:3, 3])) * 100
    if n in AX: ax = symax(n); dr = np.degrees(np.arccos(np.clip((T1[:3, :3] @ ax) @ (T2[:3, :3] @ ax), -1, 1)))
    else: dr = np.degrees(Rot.from_matrix(T1[:3, :3].T @ T2[:3, :3]).magnitude())
    return dp, dr
def ev(n, T, f):
    m = E.mask(n, 1, f)
    if m is None or m.sum() < 100: return np.nan, np.nan
    exc = E.hands(names, f)
    for q in names:
        if q != n and E.mask(q, 1, f) is not None: exc |= E.mask(q, 1, f)
    exc &= ~m; r = render_sil(E, M[n], T)
    if r is None: return 0., np.nan
    iou = (r & m & ~exc).sum() / max(1, ((r | m) & ~exc).sum()); dep = E.depth(f); zr = render_depth(E, M[n], T)
    use = m & np.isfinite(zr) & np.isfinite(dep) & ~exc
    return iou, (np.median(np.abs(dep[use] - zr[use])) * 100 if use.sum() > 50 else np.nan)
rep = dict(split=a.split, ep=a.ep, src=str(a.src), adopted=[], considered=[], ground_truth_used=False, args={k: str(v) for k, v in vars(a).items()})
for i, n in enumerate(names):
    if 'spoon' in n: continue
    cands = []
    for t, H in hyps.items():
        if n not in H: continue
        D = np.array([dis(C[f, i], H[n][f], n) for f in range(F)]); bad = (D[:, 0] > 2) | (D[:, 1] > 15); f = 0
        while f < F:
            if not bad[f]: f += 1; continue
            g = f
            while g < F and bad[g]: g += 1
            if g - f >= a.minlen:
                eo = np.array([ev(n, C[q, i], q) for q in range(f, g)]); eh = np.array([ev(n, H[n][q], q) for q in range(f, g)])
                io, ih = np.nanmean(eo[:, 0]), np.nanmean(eh[:, 0]); do, dh = np.nanmedian(eo[:, 1]), np.nanmedian(eh[:, 1])
                ok = (ih >= io + a.diou) and not (np.isfinite(dh) and np.isfinite(do) and dh > do + 0.3)
                rec = dict(obj=n, tracker=t, f0=f, f1=g - 1, pos=float(D[f:g, 0].mean()), rot=float(D[f:g, 1].mean()), iou_ours=float(io), iou_trk=float(ih), dep_ours=float(do), dep_trk=float(dh), adopt=bool(ok))
                rep['considered'].append(rec)
                if ok: cands.append((ih - io, f, g, t))
            f = g
    used = np.zeros(F, bool)
    for gain, f, g, t in sorted(cands, reverse=True):
        if used[f:g].any(): continue
        used[f:g] = True; H = hyps[t][n]
        for q in range(max(0, f - a.blend), min(F, g + a.blend)):
            w = 1.0 if f <= q < g else (1 - (f - q) / (a.blend + 1) if q < f else 1 - (q - g + 1) / (a.blend + 1))
            A, Bt = C[q, i], H[q]; R = Rot.from_matrix(A[:3, :3]) * Rot.from_rotvec(w * (Rot.from_matrix(A[:3, :3]).inv() * Rot.from_matrix(Bt[:3, :3])).as_rotvec())
            T = np.eye(4); T[:3, :3] = R.as_matrix(); T[:3, 3] = (1 - w) * A[:3, 3] + w * Bt[:3, 3]; C[q, i] = T
        rep['adopted'].append(dict(obj=n, tracker=t, f0=f, f1=g - 1, iou_gain=float(gain)))
z['camera_from_object'] = C.astype(z['camera_from_object'].dtype); z['world_from_object'] = np.einsum('tij,tnjk->tnik', cam, C).astype(z['world_from_object'].dtype)
np.savez_compressed(a.dst, **z); a.dst.with_suffix('.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep['adopted']), flush=True)
for r in rep['considered']: print('  ', json.dumps(r))
