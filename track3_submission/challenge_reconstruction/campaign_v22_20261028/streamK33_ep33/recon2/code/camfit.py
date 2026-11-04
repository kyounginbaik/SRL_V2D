"""K33-recon2 (GT-free): per-frame rigid camera inconsistency. For frame t and each reference r: static ORB matches (outside object/hand
masks, depth 0.25-1.6 m), back-projected with OUR stereo depth through the bundle world_from_rgb; RANSAC Kabsch Delta_tr (world_t -> world_r).
Induced object error = |Delta p - p| at each object's world position at t. Also stores Delta (rotvec deg, trans cm).
usage: camfit.py BUNDLE_EP_DIR OUT.tsv"""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as Rot
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camdrift.py')).read().split("if __name__")[0]
exec(src)
P = np.load(f'{B}/object_poses.npz', allow_pickle=False); Wo = P['world_from_object']; NM = list(P['names'])
def kabsch(a, b):
    ma, mb = a.mean(0), b.mean(0); U, S, Vt = np.linalg.svd((a - ma).T @ (b - mb)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, mb - R @ ma
def ransac(a, b, it=300, th=0.02):
    rng = np.random.default_rng(0); best = None
    for _ in range(it):
        i = rng.choice(len(a), 3, replace=False); R, t = kabsch(a[i], b[i]); inl = np.linalg.norm(a @ R.T + t - b, axis=1) < th
        if best is None or inl.sum() > best.sum(): best = inl
    return kabsch(a[best], b[best]) + (int(best.sum()),)
def one2(t):
    kt, dt, Dt = feats(t); out = [t]
    for r in REFS:
        kr, dr, Dr = RF[r]
        if dt is None or dr is None or r == t: out += [np.nan] * 4; continue
        ms = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dt, dr, k=2)
        g = [a for a, b in (x for x in ms if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(g) < 30: out += [np.nan] * 4; continue
        pt = np.array([kt[m.queryIdx].pt for m in g]); pr = np.array([kr[m.trainIdx].pt for m in g])
        try: F, inl = cv2.findFundamentalMat(pt, pr, cv2.FM_RANSAC, 1.5, 0.999)
        except cv2.error: inl = None
        if inl is None or inl.shape[0] != len(g): out += [np.nan] * 4; continue
        g = [m for m, i in zip(g, inl.ravel()) if i]
        zt = np.array([Dt[int(kt[m.queryIdx].pt[1]), int(kt[m.queryIdx].pt[0])] for m in g])
        zr = np.array([Dr[int(kr[m.trainIdx].pt[1]), int(kr[m.trainIdx].pt[0])] for m in g])
        a = world(kt, Dt, t, [m.queryIdx for m in g]); b = world(kr, Dr, r, [m.trainIdx for m in g])
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (zt < 1.6) & (zr < 1.6)
        if ok.sum() < 25: out += [np.nan] * 4; continue
        R, tr, n = ransac(a[ok], b[ok])
        e = [np.linalg.norm(R @ Wo[t, k][:3, 3] + tr - Wo[t, k][:3, 3]) * 100 for k in range(len(NM))]
        out += e + [float(np.degrees(Rot.from_matrix(R).magnitude())), n]
    return out
if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as Pp: rows = Pp.map(one2, range(len(Wr)))
    hdr = 'frame\t' + '\t'.join(f'{NM[0]}@{r}\t{NM[1]}@{r}\trotdeg@{r}\tn@{r}' for r in REFS)
    open(OUT, 'w').write(hdr + '\n' + '\n'.join('\t'.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) for r in rows) + '\n')
    print('done', len(rows))
