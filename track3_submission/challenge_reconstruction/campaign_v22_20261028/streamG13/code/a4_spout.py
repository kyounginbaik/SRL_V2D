"""GT-free physical cue for the ep20 pour: spout tip vs cup rim (in our camera frame) for v10 and fro (c21 only as a reported hypothesis)."""
import numpy as np, trimesh
from tracks import *; import kit
H = {t: np.load(f'../out/hyp/ep20_{t}.npz', allow_pickle=True) for t in ['v10', 'fro']}
for t, h in H.items():
    vp = trimesh.load(str(h['meshes'][1]), force='mesh').vertices; vc = trimesh.load(str(h['meshes'][0]), force='mesh').vertices
    # spout tip: farthest vertex from the centroid along the principal axis, choose the thinner end
    c = vp.mean(0); U = np.linalg.svd(vp - c, full_matrices=False)[2]; ax = U[0]; s = (vp - c) @ ax
    ends = []
    for sg in (1, -1):
        sel = vp[(sg * s) > np.percentile(sg * s, 98)]; ends.append((sel.mean(0), np.ptp(sel, 0).max(), sg))
    tip = min(ends, key=lambda e: e[1])[0] if t == 'v10' else tip
    # cup rim centre: cup axis = symmetry axis; rim = end with larger radius
    cc = vc.mean(0); Uc = np.linalg.svd(vc - cc, full_matrices=False)[2]
    import public_kit_score as K
    a = K.symmetry_axis('blue_cup', vc)[0]; sc = (vc - cc) @ a
    rims = []
    for sg in (1, -1):
        sel = vc[(sg * sc) > np.percentile(sg * sc, 95)]; r = np.linalg.norm((sel - sel.mean(0)) - np.outer((sel - sel.mean(0)) @ a, a), axis=1).mean(); rims.append((sel.mean(0), r))
    rim = max(rims, key=lambda e: e[1])[0]
    C = h['camera_from_object']
    tp = C[:, 1, :3, :3] @ tip + C[:, 1, :3, 3]; rp = C[:, 0, :3, :3] @ rim + C[:, 0, :3, 3]; d = np.linalg.norm(tp - rp, axis=1) * 100
    print(t, 'tip-rim distance cm per window:', ' '.join(f'f{a}-{b}:{d[a:b].mean():.1f}(min {d[a:b].min():.1f})' for a, b in [(0, 40), (40, 55), (55, 70), (70, 85), (85, 100), (100, 115), (115, 140)]))
