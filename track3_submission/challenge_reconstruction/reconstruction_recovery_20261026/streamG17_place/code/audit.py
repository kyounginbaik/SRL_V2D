"""G17 task 1: GT-free gross-error audit of one eval episode of a bundle (default selected20_final_v12b_g15).
Per object:
  * disagreement windows vs independent trackers (raw FoundationPose, multimap, CoTracker pointmap), in the camera frame
    (camera-path independent), after bringing our retimed track back to the RGB clock: frames > 2 cm or > 15 deg
    (cups: symmetry axis only), runs >= 3 frames; per window: silhouette IoU and stereo-depth |residual| ours vs tracker
  * half-turn twins (180 deg about each principal axis through the centroid) on 16 frames: IoU / depth vs ours
  * per-frame image QC every 2nd frame: IoU, depth median |res|; outliers IoU < 0.5 (mask >= 400 half-res px) or depth > 2 cm
Usage: audit.py EP [BUNDLE] -> out/audit/epN.json"""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import *
ep = int(sys.argv[1]); B = Path(sys.argv[2]) if len(sys.argv) > 2 else REC / 'selected20_final_v12b_g15'
bd = B / f'episode_{ep:06d}'; root = eproot('evaluation', ep); E = Episode(root)
z, names = load_poses(bd / 'object_poses.npz'); F = len(z['world_from_rgb']); mj = json.loads((bd / 'meshes.json').read_text())
meshes = {n: Mesh(bd / mj[n]['file']) for n in names}
d = delta_arr('evaluation', ep, F)
Crgb = np.stack([shift(z['camera_from_object'][:, i].astype(float), -np.asarray(d)) for i in range(len(names))], 1)
camb = z['world_from_rgb'].astype(float)
AXSYM = {'blue_cup', 'beige_cup'}
hyps = {}
for t in ['raw', 'multimap', 'pointmap_cotracker']:
    p = root / 'exports' / t / 'object_poses.npz'
    if p.exists():
        h = np.load(p); hn = [str(n) for n in h['names']]
        hyps[t] = {n: h['camera_from_object'][:F, hn.index(n)].astype(float) for n in names if n in hn}


def axis_of(n):
    w, U = np.linalg.eigh(np.cov((meshes[n].V - meshes[n].c).T)); return U  # columns: principal axes (ascending var)


def symaxis(n):
    # cups: the axis with the most distinct eigenvalue
    w, U = np.linalg.eigh(np.cov((meshes[n].V - meshes[n].c).T)); gaps = [abs(w[0] - w[1]) + abs(w[0] - w[2]), abs(w[1] - w[0]) + abs(w[1] - w[2]), abs(w[2] - w[0]) + abs(w[2] - w[1])]
    return U[:, int(np.argmax(gaps))]


def dis(T1, T2, n):
    c = meshes[n].c; dp = np.linalg.norm((T1[:3, :3] @ c + T1[:3, 3]) - (T2[:3, :3] @ c + T2[:3, 3])) * 100
    if n in AXSYM:
        ax = symaxis(n); a1, a2 = T1[:3, :3] @ ax, T2[:3, :3] @ ax; dr = np.degrees(np.arccos(np.clip(a1 @ a2, -1, 1)))
    else: dr = np.degrees(Rot.from_matrix(T1[:3, :3].T @ T2[:3, :3]).magnitude())
    return dp, dr


def img(n, T, f, dep=None):
    m = E.mask(n, 1, f)
    if m is None: return None, None
    exc = E.hands(names, f)
    for q in names:
        if q != n and E.mask(q, 1, f) is not None: exc |= E.mask(q, 1, f)
    exc &= ~m; r = render_sil(E, meshes[n], T)
    if r is None: return 0.0, None
    iou = (r & m & ~exc).sum() / max(1, ((r | m) & ~exc).sum())
    dr = None
    if dep is not None:
        zr = render_depth(E, meshes[n], T); use = m & np.isfinite(zr) & np.isfinite(dep) & ~exc
        if use.sum() > 50: dr = float(np.median(np.abs(dep[use] - zr[use])) * 100)
    return float(iou), dr


_dep = {}
def depf(f):
    if f not in _dep: _dep[f] = E.depth(f)
    return _dep[f]


def Cbc_at_rgb(i, f):
    # stereo-clock pose to pair with depth frame f (our retimed camera-frame pose at index f)
    return z['camera_from_object'][f, i].astype(float)


res = dict(ep=ep, bundle=str(B), objects={}, ground_truth_used=False)
for i, n in enumerate(names):
    o = dict(windows=[], twins={}, qc={})
    # per-frame QC (every 2nd frame)
    io, dd = [], []
    for f in range(0, F, 2):
        iou, _ = img(n, Crgb[f, i], f); _, dr = img(n, Cbc_at_rgb(i, f), f, depf(f))
        m = E.mask(n, 1, f); big = m is not None and m.sum() >= 400
        io.append((f, iou, big)); dd.append((f, dr))
    ious = [x[1] for x in io if x[1] is not None and x[2]]; drs = [x[1] for x in dd if x[1] is not None]
    o['qc'] = dict(iou_mean=float(np.mean(ious)) if ious else None, iou_p10=float(np.percentile(ious, 10)) if ious else None,
                   dep_med=float(np.median(drs)) if drs else None, dep_p90=float(np.percentile(drs, 90)) if drs else None,
                   low_iou_frames=[f for f, v, b in io if b and v is not None and v < 0.5], big_dep_frames=[f for f, v in dd if v is not None and v > 2.0])
    # disagreement windows
    for t, H in hyps.items():
        if n not in H: continue
        D = np.array([dis(Crgb[f, i], H[n][f], n) for f in range(F)])
        bad = (D[:, 0] > 2.0) | (D[:, 1] > 15.0); f = 0
        o.setdefault('tm', {})[t] = dict(pos=float(D[:, 0].mean()), rot=float(D[:, 1].mean()), frac_bad=float(bad.mean()))
        while f < F:
            if not bad[f]: f += 1; continue
            g = f
            while g < F and bad[g]: g += 1
            if g - f >= 3:
                fs = sorted(set(np.linspace(f, g - 1, min(6, g - f)).astype(int).tolist())); ev = []
                for q in fs:
                    io_o, dr_o = img(n, Crgb[q, i], q, depf(q)); io_h, dr_h = img(n, H[n][q], q, depf(q))
                    ev.append((io_o, io_h, dr_o, dr_h))
                ev = np.array([[np.nan if v is None else v for v in e] for e in ev], float)
                o['windows'].append(dict(tracker=t, f0=int(f), f1=int(g - 1), pos_mean=float(D[f:g, 0].mean()), rot_mean=float(D[f:g, 1].mean()),
                                         iou_ours=float(np.nanmean(ev[:, 0])), iou_trk=float(np.nanmean(ev[:, 1])), dep_ours=float(np.nanmean(ev[:, 2])) if np.isfinite(ev[:, 2]).any() else None,
                                         dep_trk=float(np.nanmean(ev[:, 3])) if np.isfinite(ev[:, 3]).any() else None))
            f = g
    # half-turn twins
    if n not in AXSYM:
        U = axis_of(n); c = meshes[n].c; fs = np.linspace(0, F - 1, 16).astype(int)
        for k in range(3):
            Rf = Rot.from_rotvec(np.pi * U[:, k]).as_matrix(); Tf = np.eye(4); Tf[:3, :3] = Rf; Tf[:3, 3] = c - Rf @ c
            ev = []
            for q in fs:
                a1 = img(n, Crgb[q, i], q, depf(q)); a2 = img(n, Crgb[q, i] @ Tf, q, depf(q)); ev.append((*a1, *a2))
            ev = np.array([[np.nan if v is None else v for v in e] for e in ev], float)
            o['twins'][f'axis{k}'] = dict(iou_ours=float(np.nanmean(ev[:, 0])), iou_twin=float(np.nanmean(ev[:, 2])), dep_ours=float(np.nanmean(ev[:, 1])), dep_twin=float(np.nanmean(ev[:, 3])),
                                          twin_better_iou_frames=int(np.sum(ev[:, 2] > ev[:, 0] + 0.03)), ours_better_iou_frames=int(np.sum(ev[:, 0] > ev[:, 2] + 0.03)))
    res['objects'][n] = o
    print(ep, n, json.dumps(o['qc'])[:300], '| tm', json.dumps(o.get('tm', {})), flush=True)
    for w in o['windows']: print('   win', json.dumps(w), flush=True)
    for k, v in o['twins'].items(): print('   twin', k, json.dumps(v), flush=True)
out = Path(__file__).parent.parent / 'out' / 'audit'; out.mkdir(parents=True, exist_ok=True)
(out / f'ep{ep}.json').write_text(json.dumps(res, indent=1))
