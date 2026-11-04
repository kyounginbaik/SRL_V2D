"""Geometric nesting depth of beige inside blue from radial profiles (scan meshes vs our meshes)."""
import numpy as np, trimesh, sys
F = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010/assets'
def profile(path):
    m = trimesh.load(path, force='mesh'); v = np.asarray(m.vertices); P = trimesh.sample.sample_surface(m, 80000, seed=0)[0]
    c = P.mean(0); w, U = np.linalg.eigh(np.cov((P - c).T)); g = [abs(w[i]-w[(i+1)%3]) + abs(w[i]-w[(i+2)%3]) - abs(w[(i+1)%3]-w[(i+2)%3]) for i in range(3)]; a = U[:, int(np.argmax(g))]
    h = (P - c) @ a; rvec = (P - c) - np.outer(h, a); r = np.linalg.norm(rvec, axis=1)
    bins = np.linspace(h.min(), h.max(), 60); idx = np.digitize(h, bins)
    rad = np.array([np.percentile(r[idx == i], 90) if (idx == i).sum() > 30 else np.nan for i in range(1, len(bins))]); hh = (bins[:-1] + bins[1:]) / 2
    # orient: rim = wide end at +h
    if np.nanmean(rad[-10:]) < np.nanmean(rad[:10]): hh = -hh[::-1]; rad = rad[::-1]
    hh = hh - hh.min(); return hh, rad
def nest(blue, beige, wall=0.002, clear=0.0):
    hb, rb = blue; ht, rt = beige; Hb = hb.max()
    best = 0
    for d in np.arange(0, Hb, 0.001):           # insertion depth d: beige bottom at height Hb - d in blue's frame
        z = ht + (Hb - d); ok = True
        for zi, ri in zip(z, rt):
            if zi > Hb or np.isnan(ri): continue
            rin = np.interp(zi, hb, rb) - wall
            if ri > rin + clear: ok = False; break
        if ok: best = d
        else: break
    return best
for tag, b, t in (('scan', '/data1/SRL_V2D/gt_best_per_object/blue_cup.ply', '/data1/SRL_V2D/gt_best_per_object/beige_cup.ply'), ('ours', F + '/blue_cup.obj', F + '/beige_cup.obj')):
    B, T = profile(b), profile(t)
    print(f'{tag}: blue rim/base radius {np.nanmax(B[1][-5:])*100:.2f}/{np.nanmin(B[1][:5])*100:.2f} cm, beige rim/base {np.nanmax(T[1][-5:])*100:.2f}/{np.nanmin(T[1][:5])*100:.2f} cm',
          '| max nest depth (wall 2 mm): %.1f cm, (wall 3 mm): %.1f cm' % (nest(B, T, .002) * 100, nest(B, T, .003) * 100))
