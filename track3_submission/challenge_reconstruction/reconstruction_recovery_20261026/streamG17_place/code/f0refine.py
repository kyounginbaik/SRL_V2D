"""G17 joint multi-object frame-0 (static-window) placement refinement. GT-free.
Per object i a constant world-frame correction D_i (rotation about the object's centroid + translation) is fitted over the
object's own static window at the start of the clip (frames where our track stays within 0.8 cm / 4 deg of frame 0),
jointly for all objects, using:
  sil   silhouette term vs our SAM2 masks (RGB clock camera); mode 'iou' (1-IoU) or 'mom' (visible-region centroid +
        trace-normalised second moments: invariant to a uniform mask erosion, so it constrains lateral placement and
        in-image rotation only)
  rdep  relative stereo depth (stereo clock camera): per frame, the median (stereo - rendered) depth on each object's
        visible pixels, minus the mean over objects; only RELATIVE along-ray placement is constrained (stereo has a
        common toward-camera bias)
  rsup  relative support: objects resting on the table (lowest hull point within 3 cm of the stereo table plane at
        start) should share the same height offset above the plane (absolute offset is a nuisance)
  prior |t|/s_t, |r|/s_r
Usage: f0refine.py SPLIT EP POSES_NPZ(RGB clock, pre-retime) MESHES_JSON OUT_JSON [--sil iou|mom] [--wd W] [--ws W] ..."""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import *

ap = argparse.ArgumentParser()
ap.add_argument('split'); ap.add_argument('ep', type=int); ap.add_argument('poses', type=Path); ap.add_argument('meshes', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--sil', default='mom'); ap.add_argument('--wsil', type=float, default=1.0); ap.add_argument('--wd', type=float, default=1.0); ap.add_argument('--ws', type=float, default=1.0)
ap.add_argument('--st', type=float, default=1.5); ap.add_argument('--sr', type=float, default=5.0)
ap.add_argument('--win_t', type=float, default=0.8); ap.add_argument('--win_r', type=float, default=4.0); ap.add_argument('--maxwin', type=int, default=60)
ap.add_argument('--nf', type=int, default=8); ap.add_argument('--rot', type=int, default=1)
a = ap.parse_args(); t0 = time.time()
root = eproot(a.split, a.ep); E = Episode(root)
z, names = load_poses(a.poses); Wt = z['world_from_object'].astype(np.float64); cam = z['world_from_rgb'].astype(np.float64); F = len(cam)
mj = json.loads(a.meshes.read_text())
meshes = {n: Mesh(mj[n]['file'] if isinstance(mj[n], dict) else mj[n]) for n in names}
d = delta_arr(a.split, a.ep, F); cam_bc = shift(cam, d)   # camera at stereo-clock frame g (world static objects)
# static windows
L = {}
for i, n in enumerate(names):
    T0 = Wt[0, i]; l = 1
    while l < min(F, a.maxwin):
        dt = np.linalg.norm(Wt[l, i, :3, 3] - T0[:3, 3]) * 100; dr = np.degrees(Rot.from_matrix(T0[:3, :3].T @ Wt[l, i, :3, :3]).magnitude())
        if dt > a.win_t or dr > a.win_r: break
        l += 1
    L[n] = l
Lm = max(L.values()); frames = sorted(set(np.linspace(0, Lm - 1, min(a.nf, Lm)).astype(int).tolist()))
res = dict(split=a.split, ep=a.ep, poses=str(a.poses), names=names, windows=L, frames=frames, args=vars(a) | dict(poses=str(a.poses), meshes=str(a.meshes), out=str(a.out)), ground_truth_used=False)
if len(names) < 2:
    res.update(status='single object: frame-0 alignment absorbs any placement; no change', D={n: np.eye(4).tolist() for n in names})
    a.out.write_text(json.dumps(res, indent=1)); raise SystemExit(0)
cen = {n: Wt[0, i, :3, :3] @ meshes[n].c + Wt[0, i, :3, 3] for i, n in enumerate(names)}
# preload per frame
data = []
for f in frames:
    hands = E.hands(names, f); ms = {n: E.mask(n, 1, f) for n in names}
    dep = E.depth(f) if a.wd > 0 else None
    data.append((f, hands, ms, dep))
pl = E.plane_world(cam_bc, list(range(Lm)))
inv = np.linalg.inv


def obj_pose(D, i, f, camera):
    return inv(camera[f]) @ D @ Wt[f, i]


def moments(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 30: return None
    c = np.array([xs.mean(), ys.mean()]); X = np.stack([xs - c[0], ys - c[1]], 1); C = X.T @ X / len(xs)
    return c, C / np.trace(C)


def terms(p, detail=False):
    Ds = {n: about(cen[n], p[6 * i:6 * i + 3] * (a.rot > 0), p[6 * i + 3:6 * i + 6] / 100) for i, n in enumerate(names)}
    sil, dres = [], []
    for f, hands, ms, dep in data:
        act = [n for n in names if f < L[n]]
        rs = {}; zs = {}
        for i, n in enumerate(names):
            rs[n] = render_sil(E, meshes[n], obj_pose(Ds[n] if f < L[n] else np.eye(4), i, f, cam))
        for i, n in enumerate(act):
            m = ms[n]; r = rs[n]
            if m is None or r is None or m.sum() < 200: continue
            exc = hands.copy()
            for o in names:
                if o != n:
                    if ms[o] is not None: exc |= ms[o]
            exc &= ~m
            if a.sil == 'iou':
                u = (r | m) & ~exc; sil.append(1 - (r & m & ~exc).sum() / max(1, u.sum()))
            else:
                # visible rendered region: rendered minus pixels covered by occluders (hands/other masks)
                mo, ro = moments(m & ~exc), moments(r & ~exc)
                if mo is None or ro is None: sil.append(25.0); continue
                sil.append(np.sum((mo[0] - ro[0]) ** 2) / 2.0 ** 2 + np.sum((mo[1] - ro[1]) ** 2) / 0.02 ** 2)
        if dep is not None:
            ms_ = {}
            for i, n in enumerate(act):
                zr = render_depth(E, meshes[n], obj_pose(Ds[n], names.index(n), f, cam_bc))
                m = ms[n]
                if zr is None or m is None: continue
                use = m & np.isfinite(zr) & np.isfinite(dep) & ~hands
                for o in names:
                    if o != n and ms[o] is not None: use &= ~ms[o]
                if use.sum() < 100: continue
                ms_[n] = np.median(dep[use] - zr[use]) * 100
            if len(ms_) >= 2:
                v = np.array(list(ms_.values())); dres.extend(((v - v.mean()) / 0.5).tolist())
    sup = []
    if pl is not None and a.ws > 0:
        nw, dw = pl; hs = []
        for i, n in enumerate(names):
            T = Ds[n] @ Wt[0, i]; h = (meshes[n].Vh @ T[:3, :3].T + T[:3, 3]) @ nw + dw
            hs.append(h.min() * 100)
        hs = np.array(hs); rest = np.abs(hs0) < 4.0
        if rest.sum() >= 2: sup = ((hs[rest] - hs[rest].mean()) / 0.5).tolist()
    prior = sum(np.sum((p[6 * i + 3:6 * i + 6] / a.st) ** 2) + np.sum((np.degrees(p[6 * i:6 * i + 3]) / a.sr) ** 2) * (a.rot > 0) for i in range(len(names)))
    t = dict(sil=float(np.mean(sil)) if sil else 0., rdep=float(np.mean(np.square(dres))) if dres else 0., rsup=float(np.mean(np.square(sup))) if sup else 0., prior=float(prior),
             n_sil=len(sil), n_dep=len(dres), n_sup=len(sup))
    t['sil_raw'] = sil if detail else None
    t['total'] = a.wsil * t['sil'] / (0.02 if a.sil == 'iou' else 1.0) + a.wd * t['rdep'] + a.ws * t['rsup'] + t['prior']
    return t


hs0 = None
if pl is not None:
    nw, dw = pl; hs0 = np.array([(((meshes[n].Vh @ Wt[0, i, :3, :3].T + Wt[0, i, :3, 3]) @ nw + dw).min() * 100) for i, n in enumerate(names)])
    hs0 = hs0 if (np.abs(hs0) < 3.0).sum() >= 2 else hs0
    res['heights_cm_before'] = hs0.tolist(); res['plane'] = dict(n=pl[0].tolist(), d=pl[1])
p0 = np.zeros(6 * len(names)); before = terms(p0)
steps = np.tile([np.radians(1.5)] * 3 + [0.5] * 3, len(names))
r = minimize(lambda q: terms(q)['total'], p0, method='Powell', options=dict(xtol=1e-3, ftol=1e-4, maxfev=2500, direc=np.diag(steps)))
p = r.x; after = terms(p)
Ds = {n: about(cen[n], p[6 * i:6 * i + 3] * (a.rot > 0), p[6 * i + 3:6 * i + 6] / 100) for i, n in enumerate(names)}
res.update(before={k: v for k, v in before.items() if k != 'sil_raw'}, after={k: v for k, v in after.items() if k != 'sil_raw'},
           D={n: Ds[n].tolist() for n in names}, params={n: dict(rv=(p[6 * i:6 * i + 3] * (a.rot > 0)).tolist(), t=(p[6 * i + 3:6 * i + 6] / 100).tolist(), c=cen[n].tolist()) for i, n in enumerate(names)},
           corr={n: dict(t_cm=p[6 * i + 3:6 * i + 6].tolist(), rot_deg=float(np.degrees(np.linalg.norm(p[6 * i:6 * i + 3]))) * (a.rot > 0)) for i, n in enumerate(names)},
           nfev=int(r.nfev), seconds=time.time() - t0)
# relative placement change between objects at frame 0 (what the frame-0 scene alignment sees)
rel = {}
for i, n in enumerate(names):
    for j, m in enumerate(names):
        if j <= i: continue
        A0 = inv(Wt[0, i]) @ Wt[0, j]; A1 = inv(Ds[n] @ Wt[0, i]) @ Ds[m] @ Wt[0, j]
        rel[f'{n}->{m}'] = dict(dt_cm=float(np.linalg.norm(A1[:3, 3] - A0[:3, 3]) * 100), dr_deg=float(np.degrees(Rot.from_matrix(A0[:3, :3].T @ A1[:3, :3]).magnitude())))
res['rel_change'] = rel
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
print(json.dumps(dict(ep=a.ep, windows=L, before=res['before'], after=res['after'], corr=res['corr'], rel=rel, sec=round(res['seconds']))), flush=True)
