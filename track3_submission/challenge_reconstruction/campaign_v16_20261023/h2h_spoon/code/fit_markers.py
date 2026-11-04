"""Track-consistency test with the mocap balls (depth- and mask-free): for each track, triangulate the K ball positions in the
object body frame from the detected ball pixels through that track's per-frame poses (rays in its own camera model), then measure
the reprojection residual of every detection (px, full resolution). A correct track explains every detection with one constant
set of body points; roll flips / along-handle slips show up as large residuals in their windows. Also reports the residual of the
180-deg roll twin (R * Rx(180)) with the same body points.
usage: fit_markers.py EP OBJ BALLS.json OUT.json [K]"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H, cammodel as CM
from scipy.spatial.transform import Rotation as R
ep, obj, bj, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; K = int(sys.argv[5]) if len(sys.argv) > 5 else 3
e = H.Ep(ep, nsamp=10); B = json.load(open(bj))['rows']
dets = [(r['frame'], np.array(d['px'])) for r in B for d in r['dets']]
F = np.array([f for f, _ in dets]); UV = np.array([p for _, p in dets]) if dets else np.zeros((0, 2))
TW = np.eye(4); TW[1, 1] = TW[2, 2] = -1   # Rx(180)
def model_of(track): return e.model_of(track)
def fit(track, twin=False):
    m = model_of(track); rays = CM.rays_at(e.cm, m, UV); C = np.array([e.cam_T(track, obj, f) for f in range(e.N)])
    if twin: C = C @ TW
    Rf, tf = C[F, :3, :3], C[F, :3, 3]; d = np.einsum('nji,nj->ni', Rf, rays); d /= np.linalg.norm(d, axis=1, keepdims=True); o = -np.einsum('nji,nj->ni', Rf, tf)
    # init: lift by depth (near percentile in a 5x5 window) -> body points -> 1D k-means along body x
    zs = []
    for f, p in zip(F, UV):
        dep = e.depth(int(f)) if not hasattr(fit, 'dc') or fit.dc[0] != f else fit.dc[1]; fit.dc = (f, dep)
        u, v = int(p[0] * e.cm.sc), int(p[1] * e.cm.sc); w = dep[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]; w = w[np.isfinite(w)]
        zs.append(np.percentile(w, 30) + 0.006 if len(w) else np.nan)
    zs = np.array(zs); Pc = rays * zs[:, None]; Pb = np.einsum('nji,nj->ni', Rf, Pc - tf); ok = np.isfinite(zs) & (np.abs(Pb).max(1) < 0.25)
    c = np.percentile(Pb[ok, 0], np.linspace(10, 90, K)) if ok.sum() > K else np.linspace(-.1, .1, K)
    for _ in range(30):
        lab = np.argmin(np.abs(Pb[ok, 0][:, None] - c[None]), 1); c = np.array([np.median(Pb[ok][lab == j, 0]) if (lab == j).any() else c[j] for j in range(K)])
    balls = np.array([np.median(Pb[ok][lab == j], 0) if (lab == j).sum() > 0 else np.array([c[j], 0, 0]) for j in range(K)])
    def reproj(balls):
        X = np.einsum('nij,kj->nki', Rf, balls) + tf[:, None]; uv = np.stack([CM.project_full(e.cm, m, X[:, k])[0] for k in range(K)], 1)
        dist = np.linalg.norm(uv - UV[:, None], axis=2); return dist
    for it in range(15):
        dist = reproj(balls); a = dist.argmin(1); r = dist.min(1); gate = max(np.percentile(r, 60), 12.0)
        for j in range(K):
            s = (a == j) & (r < gate * 2.5)
            if s.sum() < 5: continue
            A = np.zeros((3, 3)); b = np.zeros(3)
            for dd, oo in zip(d[s], o[s]): P = np.eye(3) - np.outer(dd, dd); A += P; b += P @ oo
            balls[j] = np.linalg.solve(A + 1e-9 * np.eye(3), b)
    dist = reproj(balls); return balls, dist.min(1), dist.argmin(1)
res = {}
for track in ('ours', 'new'):
    balls, r, a = fit(track); res[track] = dict(balls_cm=(balls * 100).round(2).tolist(), resid_px=r.round(2).tolist(), ball=a.tolist())
    # twin with the same balls
    m = model_of(track); C = np.array([e.cam_T(track, obj, f) for f in range(e.N)]) @ TW; Rf, tf = C[F, :3, :3], C[F, :3, 3]
    X = np.einsum('nij,kj->nki', Rf, balls) + tf[:, None]; uv = np.stack([CM.project_full(e.cm, m, X[:, k])[0] for k in range(len(balls))], 1)
    res[track]['twin_resid_px'] = np.linalg.norm(uv - UV[:, None], axis=2).min(1).round(2).tolist()
    print(f'ep{ep} {obj} {track}: balls (cm) {np.round(balls * 100, 1).tolist()}  resid median {np.median(r):.1f} px, <15px {np.mean(r < 15):.2f}, twin median {np.median(res[track]["twin_resid_px"]):.1f}')
json.dump(dict(ep=ep, obj=obj, frames=F.tolist(), uv=UV.round(1).tolist(), **res), open(out, 'w'))
# per-second table
print(' sec | n  | ours med / <10 / <20px | new med / <10 / <20px | twin<20 ours / new')
for s0 in range(0, e.N, 20):
    sel = (F >= s0) & (F < s0 + 20)
    if sel.sum() == 0: continue
    ro, rn = np.array(res['ours']['resid_px'])[sel], np.array(res['new']['resid_px'])[sel]
    to, tn = np.array(res['ours']['twin_resid_px'])[sel], np.array(res['new']['twin_resid_px'])[sel]
    print(f'{s0/20:4.0f} | {sel.sum():3d} | {np.median(ro):6.1f} / {np.mean(ro<10):.2f} / {np.mean(ro<20):.2f} | {np.median(rn):6.1f} / {np.mean(rn<10):.2f} / {np.mean(rn<20):.2f} | {np.mean(to<20):.2f} / {np.mean(tn<20):.2f}')
