"""ep36 tgt_s4 step 1+2 (GT-free, no human marks): on tgt_s1, (1) replace f297-308 by a position spline + rotation slerp from the
neighbouring good frames (FINAL depth/position jump), (2) depth-locked silhouette fit for 12.5-17.5 s: per frame only a camera-plane shift
(camera z fixed) and a rotation about the viewing ray through the spoon COM; corrections smoothed (sigma 3 frames) with 0.25 s ramps.
Writes WORK/T4a.npz and a roll-fit bundle WORK/bundle/episode_000036 (FINAL v14_g17 bundle with the spoon right-multiplied by
inv(T_final) @ T4a). usage: s4pos.py WORK"""
import sys, json, shutil, numpy as np, cv2, trimesh
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

WORK = Path(sys.argv[1]); WORK.mkdir(parents=True, exist_ok=True)
E = 'episode_000036'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
COMP = 'streamALL20K/staging/tgt_v32_ep36_b2/components/episode_000036'; TGT = C22 / 'target_final_v14_g17' / E
S1 = Path('/data1/hansol/spoon3537/targets/tgt_s1') / E
BUN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
man = json.loads((C22 / COMP / f'portable_validation/runtime/episodes/{E}/input_manifest.json').read_text()); SRC = Path(man['source'])
SAMO = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs') / E / 'wooden_spoon'
op = np.load(SRC / 'object_poses.npz'); si = [str(n) for n in op['names']].index('wooden_spoon')
Wold = op['world_from_object'][:, si].astype(np.float64); Wrgb = op['world_from_rgb'].astype(np.float64)
Told = np.load(TGT / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); N = len(Told)
T1 = np.load(S1 / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64)
Sx = np.einsum('tij,tjk->tik', Told, np.linalg.inv(Wold)); fps = float(man['fps'])
tocam = lambda T: np.einsum('tij,tjk->tik', np.linalg.inv(Wrgb), np.einsum('tij,tjk->tik', np.linalg.inv(Sx), T))
totgt = lambda C: np.einsum('tij,tjk->tik', Sx, np.einsum('tij,tjk->tik', Wrgb, C))
m = trimesh.load(SRC / 'assets/wooden_spoon.obj', force='mesh', process=False); V = np.asarray(m.vertices); F = np.asarray(m.faces); COM = V.mean(0)
d = json.loads((SAMO / 'anycalib/undistorted_intrinsics.json').read_text()); SC = 0.5
K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1.]]); WH = (int(d['width']), int(d['height']))
aw, ah = int(WH[0] * SC), int(WH[1] * SC); Ks = K.copy(); Ks[:2] *= SC
J0, J1 = 297, 308; W0, W1 = 12.5, 17.5


def sil(T):
    p = V @ T[:3, :3].T + T[:3, 3]; z = p[:, 2]; img = np.zeros((ah, aw), np.uint8)
    if (z < 0.02).any(): return img > 0
    uv = p[:, :2] / z[:, None] @ Ks[:2, :2].T + Ks[:2, 2]
    cv2.fillPoly(img, list(np.round(uv[F] * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2); return img > 0


def load(t):
    rd = lambda o: cv2.resize(cv2.imread(str(SAMO / f'masks/{o}/{t:06d}.png'), cv2.IMREAD_GRAYSCALE), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
    try: return rd(1), rd(2) | rd(3)
    except Exception: return None, None


def score(T, Mk, h):
    S = sil(T); a = S & ~h; b = Mk & ~h; iou = (a & b).sum() / max((a | b).sum(), 1); sd = ((a & ~b).sum() + (b & ~a).sum()) / max(b.sum(), 1)
    ys, xs = np.nonzero(b); ya, xa = np.nonzero(a)
    cen = float(np.hypot(xs.mean() - xa.mean(), ys.mean() - ya.mean())) / SC if len(xs) and len(xa) else np.nan
    return sd, float(iou), cen, int(S.sum())


def app(T, x):   # x = [rotation about the viewing ray (0.1 rad), dx, dy (cm, camera plane, z fixed)]
    c = T[:3, :3] @ COM + T[:3, 3]; r = c / np.linalg.norm(c); Q = R.from_rotvec(r * x[0] * 0.1).as_matrix()
    O = T.copy(); O[:3, :3] = Q @ T[:3, :3]; O[:3, 3] = Q @ (T[:3, 3] - c) + c + np.array([x[1], x[2], 0.0]) * 0.01; return O


def cost(x, T0, Mk, h):
    sd = score(app(T0, x), Mk, h)[0]; return sd + 0.08 * (np.degrees(abs(x[0]) * 0.1) / 15) ** 2 + 0.002 * (x ** 2).sum()


CAMb = None


def fit(t):
    Mk, h = load(t)
    if Mk is None: return t, None, None, None, 0
    vis = int((Mk & ~h).sum()); T0 = CAMb[t]; s0 = score(T0, Mk, h)
    if vis < 150: return t, T0, s0, s0, vis
    starts = [np.zeros(3)]; a = sil(T0) & ~h; b = Mk & ~h
    if a.sum() and b.sum():
        ya, xa = np.nonzero(a); yb, xb = np.nonzero(b); z = (T0[:3, :3] @ COM + T0[:3, 3])[2]
        starts.append(np.array([0, (xb.mean() - xa.mean()) / Ks[0, 0] * z * 100, (yb.mean() - ya.mean()) / Ks[1, 1] * z * 100]))
    best = None
    for x0 in starts:
        r = minimize(cost, x0, args=(T0, Mk, h), method='Powell', bounds=[(-5, 5), (-12, 12), (-12, 12)], options=dict(maxfev=500, xtol=0.01, ftol=1e-4))
        if best is None or r.fun < best[0]: best = (r.fun, r.x)
    T1_ = app(T0, best[1]); return t, T1_, s0, score(T1_, Mk, h), vis


if __name__ == '__main__':
    # (1) jump removal on tgt_s1
    Tj = T1.copy(); good = [t for t in list(range(J0 - 10, J0)) + list(range(J1 + 1, J1 + 11))]
    cs = CubicSpline(good, T1[good, :3, 3], axis=0); sl = Slerp([J0 - 1, J1 + 1], R.from_matrix(np.stack([T1[J0 - 1, :3, :3], T1[J1 + 1, :3, :3]])))
    for t in range(J0, J1 + 1): Tj[t, :3, 3] = cs(t); Tj[t, :3, :3] = sl(t).as_matrix()
    CAMb = tocam(Tj)
    fr = [t for t in range(N) if (W0 - 0.3) * fps <= t <= (W1 + 0.3) * fps]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as P: res = P.map(fit, fr, chunksize=2)
    dp = np.zeros((N, 3)); dr = np.zeros((N, 3)); wt = np.zeros(N)
    for t, Tn, s0, s1, vis in res:
        if Tn is None or s0 is None or vis < 150 or s1[1] < 0.25 or s1[1] - s0[1] < -0.02: continue
        T0 = CAMb[t]; dp[t] = (Tn[:3, :3] @ COM + Tn[:3, 3]) - (T0[:3, :3] @ COM + T0[:3, 3]); dr[t] = R.from_matrix(Tn[:3, :3] @ T0[:3, :3].T).as_rotvec()
        wt[t] = min(1.0, vis / 600) * (0.5 + s1[1])
    k = np.exp(-0.5 * (np.arange(-10, 11) / 3.0) ** 2); den = np.convolve(wt, k, 'same'); sp = np.where(den > 0.3)[0]
    sdp = np.zeros((N, 3)); sdr = np.zeros((N, 3))
    for i in range(3):
        sdp[:, i] = np.interp(np.arange(N), sp, np.convolve(dp[:, i] * wt, k, 'same')[sp] / den[sp]); sdr[:, i] = np.interp(np.arange(N), sp, np.convolve(dr[:, i] * wt, k, 'same')[sp] / den[sp])
    tt = np.arange(N) / fps; w = np.clip(np.minimum((tt - W0) / 0.25 + 1, (W1 - tt) / 0.25 + 1), 0, 1); w[0] = 0
    C4 = CAMb.copy()
    for t in np.where(w > 0)[0]:
        T0 = CAMb[t]; c0 = T0[:3, :3] @ COM + T0[:3, 3]; Q = R.from_rotvec(sdr[t] * w[t]).as_matrix()
        C4[t, :3, :3] = Q @ T0[:3, :3]; C4[t, :3, 3] = Q @ (T0[:3, 3] - c0) + c0 + sdp[t] * w[t]
    T4a = totgt(C4); assert np.allclose(T4a[0], Told[0])
    np.savez(WORK / 'T4a.npz', T4a=T4a, Tj=Tj, w=w, wt=wt, sdp=sdp, sdr=sdr)
    zc = lambda C, t: float((C[t, :3, :3] @ COM + C[t, :3, 3])[2])
    print('camera z of spoon COM (cm) at 15.15 s: tgt_s1 %.1f  jump-fixed %.1f  T4a %.1f; neighbours f296 %.1f f310 %.1f' % (
        100 * zc(tocam(T1), 303), 100 * zc(CAMb, 303), 100 * zc(C4, 303), 100 * zc(CAMb, 296), 100 * zc(CAMb, 310)))
    # roll bundle: FINAL bundle with the spoon right-multiplied by inv(T_final) @ T4a
    Rel = np.einsum('tij,tjk->tik', np.linalg.inv(Told), T4a)
    bd = WORK / 'bundle' / E
    if bd.exists(): shutil.rmtree(bd)
    shutil.copytree(BUN, bd); z = dict(np.load(BUN / 'object_poses.npz')); s = [str(n) for n in z['names']].index('wooden_spoon')
    for key in ('world_from_object', 'camera_from_object'):
        a = z[key].copy(); a[:, s] = np.einsum('tij,tjk->tik', a[:, s], Rel); z[key] = a
    np.savez_compressed(bd / 'object_poses.npz', **z); print('BUNDLE', bd)
