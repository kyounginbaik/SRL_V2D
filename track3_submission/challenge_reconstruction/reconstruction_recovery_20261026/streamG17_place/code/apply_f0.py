"""Apply G17 frame-0 placement corrections (f0refine.py JSON) to a (retimed) pose npz.
mode 'win': correction on the object's static window (window shifted by the clock delta), linearly blended to identity over
--blend frames after it; 'all': the whole track. camera_from_object is recomputed from world_from_rgb.
Usage: apply_f0.py SRC_NPZ REFINE_JSON DST_NPZ [--mode win|all] [--blend 10] [--objs a,b] [--scale 1.0]"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import about, delta_arr


def apply(z, R, mode='win', blend=10, objs=None, scale=1.0):
    z = dict(z); names = [str(n) for n in z['names']]; W = z['world_from_object'].astype(np.float64).copy(); F = len(W)
    if 'params' not in R: return z, {}
    d = delta_arr(R['split'], R['ep'], F); dd = float(np.min(d)) if np.ndim(d) else d
    info = {}
    for i, n in enumerate(names):
        if objs and n not in objs: continue
        p = R['params'][n]; rv, t, c = np.array(p['rv']) * scale, np.array(p['t']) * scale, np.array(p['c'])
        if mode == 'all': w = np.ones(F)
        else:
            Lw = max(1, int(round(R['windows'][n] - dd)))
            w = np.clip(1 - (np.arange(F) - (Lw - 1)) / (blend + 1), 0, 1)
        for f in range(F):
            if w[f] > 0: W[f, i] = about(c, w[f] * rv, w[f] * t) @ W[f, i]
        info[n] = dict(mode=mode, frames_touched=int((w > 0).sum()), t_cm=(t * 100).tolist(), rot_deg=float(np.degrees(np.linalg.norm(rv))))
    z['world_from_object'] = W.astype(z['world_from_object'].dtype)
    if 'camera_from_object' in z:
        z['camera_from_object'] = np.einsum('tij,tnjk->tnik', np.linalg.inv(z['world_from_rgb'].astype(np.float64)), W).astype(z['camera_from_object'].dtype)
    return z, info


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src', type=Path); ap.add_argument('refine', type=Path); ap.add_argument('dst', type=Path)
    ap.add_argument('--mode', default='win'); ap.add_argument('--blend', type=int, default=10); ap.add_argument('--objs'); ap.add_argument('--scale', type=float, default=1.0)
    a = ap.parse_args()
    if a.dst.exists(): raise SystemExit('exists; never overwrite')
    z, info = apply(np.load(a.src), json.loads(a.refine.read_text()), a.mode, a.blend, a.objs.split(',') if a.objs else None, a.scale)
    np.savez_compressed(a.dst, **z); print(json.dumps(info))
