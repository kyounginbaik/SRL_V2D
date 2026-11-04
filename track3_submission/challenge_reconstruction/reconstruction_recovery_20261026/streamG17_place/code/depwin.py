import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import *
ep, n, f0, f1 = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); B = REC / 'selected20_final_v12b_g15'
bd = B / f'episode_{ep:06d}'; E = Episode(eproot('evaluation', ep)); z, names = load_poses(bd / 'object_poses.npz'); mj = json.loads((bd / 'meshes.json').read_text())
i = names.index(n); Ms = Mesh(bd / mj[n]['file']); F = len(z['world_from_rgb']); d = delta_arr('evaluation', ep, F)
C = z['camera_from_object'][:, i].astype(float); Crgb = shift(C, -np.asarray(d))
hy = {t: np.load(E.root / 'exports' / t / 'object_poses.npz') for t in ['raw', 'multimap', 'pointmap_cotracker']}
for f in range(f0, f1):
    dep = E.depth(f); m = E.mask(n, 1, f); exc = E.hands(names, f)
    for q in names:
        if q != n and E.mask(q, 1, f) is not None: exc |= E.mask(q, 1, f)
    exc &= ~m; row = [f'{f:4d} mask {m.sum():5d} hand {E.hands(names,f).sum():5d}']
    for lab, T, Tb in [('ours', Crgb[f], C[f])] + [(t[:3], h['camera_from_object'][f, [str(x) for x in h['names']].index(n)].astype(float), shift(h['camera_from_object'][:, [str(x) for x in h['names']].index(n)].astype(float), d)[f]) for t, h in hy.items()]:
        r = render_sil(E, Ms, T); iou = (r & m & ~exc).sum() / max(1, ((r | m) & ~exc).sum()); zr = render_depth(E, Ms, Tb); use = m & np.isfinite(zr) & np.isfinite(dep) & ~exc
        rr = (dep[use] - zr[use]) * 100 if use.sum() > 50 else np.array([np.nan])
        row.append(f'{lab} {iou:.2f} {np.median(rr):+.1f} z{T[2,3]*100:.0f}')
    print(' | '.join(row), flush=True)
