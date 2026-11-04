"""GT-free: signed stereo depth residual (stereo - rendered, cm, median over visible object pixels) and IoU per frame for one eval episode of a bundle. Usage: depcheck.py EP STEP [BUNDLE]"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import *
ep, step = int(sys.argv[1]), int(sys.argv[2]); B = Path(sys.argv[3]) if len(sys.argv) > 3 else REC / 'selected20_final_v12b_g15'
bd = B / f'episode_{ep:06d}'; E = Episode(eproot('evaluation', ep)); z, names = load_poses(bd / 'object_poses.npz'); mj = json.loads((bd / 'meshes.json').read_text())
M = {n: Mesh(bd / mj[n]['file']) for n in names}; F = len(z['world_from_rgb']); d = delta_arr('evaluation', ep, F)
Crgb = np.stack([shift(z['camera_from_object'][:, i].astype(float), -np.asarray(d)) for i in range(len(names))], 1)
for f in range(0, F, step):
    dep = E.depth(f); hands = E.hands(names, f); row = [f'{f:4d}']
    for i, n in enumerate(names):
        m = E.mask(n, 1, f); exc = hands.copy()
        for q in names:
            if q != n and E.mask(q, 1, f) is not None: exc |= E.mask(q, 1, f)
        if m is None: row.append(f'{n[:8]} nomask'); continue
        exc &= ~m; r = render_sil(E, M[n], Crgb[f, i]); iou = (r & m & ~exc).sum() / max(1, ((r | m) & ~exc).sum())
        zr = render_depth(E, M[n], z['camera_from_object'][f, i].astype(float)); use = m & np.isfinite(zr) & np.isfinite(dep) & ~exc
        rr = (dep[use] - zr[use]) * 100 if use.sum() > 50 else np.array([np.nan])
        row.append(f'{n[:8]} IoU {iou:.2f} dres med {np.median(rr):+.2f} p25 {np.percentile(rr,25):+.2f} p75 {np.percentile(rr,75):+.2f} npx {use.sum()} zc {z["camera_from_object"][f,i,2,3]*100:.1f}')
    print(' | '.join(row), flush=True)
