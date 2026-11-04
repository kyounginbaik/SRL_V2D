"""GT-free spoon refit for single-spoon episodes (35/36/37), no human marks.
Per frame in the given windows: 6-DoF correction of the current target's spoon pose (camera frame, about the COM) that best matches the
SAM2 spoon mask (hand masks 2,3 excluded) with priors to stay near the old pose (rotation, depth). Corrections are then smoothed in time
(weighted Gaussian, invalid frames weight 0) and ramped in/out (RAMP s) at the window edges, so the old pose's motion is kept and only the
slow bias is removed. Optional jump repair: frames whose old speed exceeds 100 cm/s or 300 deg/s inside JUMP windows are replaced by
interpolation between the nearest good frames. Frame 0 is never changed.
usage: fit_spoon.py EP OUTROOT WINDOWS(a:b,c:d s) [JUMP windows or -] [workers]"""
import sys, json, shutil, subprocess, numpy as np, cv2, trimesh
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R, Slerp

EP = int(sys.argv[1]); OUTROOT = Path(sys.argv[2]); WINS = [tuple(map(float, w.split(':'))) for w in sys.argv[3].split(',')]
JUMPS = [] if len(sys.argv) <= 4 or sys.argv[4] == '-' else [tuple(map(float, w.split(':'))) for w in sys.argv[4].split(',')]
WK = int(sys.argv[5]) if len(sys.argv) > 5 else 32
E = f'episode_{EP:06d}'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
COMP = {35: 'streamK35_ep35/staging/K35_H2joint_b1/components/episode_000035', 36: 'streamALL20K/staging/tgt_v32_ep36_b2/components/episode_000036',
        37: 'streamALL20K/staging/tgt_v32_ep37_b1/components/episode_000037'}[EP]
TGT = {35: C22 / 'streamK35_ep35/targets/K35_H2joint/episode_000035', 36: C22 / 'target_final_v14_g17/episode_000036', 37: C22 / 'target_final_v14_g17/episode_000037'}[EP]
man = json.loads((C22 / COMP / f'portable_validation/runtime/episodes/{E}/input_manifest.json').read_text()); SRC = Path(man['source'])
SAMO = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs') / E / 'wooden_spoon'
op = np.load(SRC / 'object_poses.npz'); si = [str(n) for n in op['names']].index('wooden_spoon')
Wold = op['world_from_object'][:, si].astype(np.float64); Wrgb = op['world_from_rgb'].astype(np.float64)
Told = np.load(TGT / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); N = len(Told); assert len(Wold) == N
Sx = np.einsum('tij,tjk->tik', Told, np.linalg.inv(Wold))   # target frame <- source world, per frame
fps = float(man['fps'])
m = trimesh.load(SRC / 'assets/wooden_spoon.obj', force='mesh', process=False); V = np.asarray(m.vertices); F = np.asarray(m.faces); COM = V.mean(0)
d = json.loads((SAMO / 'anycalib/undistorted_intrinsics.json').read_text()); SC = 0.5
K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1.]]); WH = (int(d['width']), int(d['height']))
aw, ah = int(WH[0] * SC), int(WH[1] * SC); Ks = K.copy(); Ks[:2] *= SC
CAMold = np.einsum('tij,tjk->tik', np.linalg.inv(Wrgb), Wold)


def sil(T):
    p = V @ T[:3, :3].T + T[:3, 3]; z = p[:, 2]; img = np.zeros((ah, aw), np.uint8)
    if (z < 0.02).any(): return img > 0
    uv = p[:, :2] / z[:, None] @ Ks[:2, :2].T + Ks[:2, 2]
    cv2.fillPoly(img, list(np.round(uv[F] * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2); return img > 0


def load(t):
    rd = lambda o: cv2.resize(cv2.imread(str(SAMO / f'masks/{o}/{t:06d}.png'), cv2.IMREAD_GRAYSCALE), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
    try:
        Mk = rd(1); h = rd(2) | rd(3)
    except Exception:
        return None, None
    return Mk, h


def app(T, x):
    O = T.copy(); Q = R.from_rotvec(x[:3] * 0.1).as_matrix(); c = T[:3, :3] @ COM + T[:3, 3]
    O[:3, :3] = Q @ T[:3, :3]; O[:3, 3] = Q @ (T[:3, 3] - c) + c + x[3:] * 0.01; return O


def score(T, Mk, h):
    S = sil(T); a = S & ~h; b = Mk & ~h; u = (a | b).sum(); iou = (a & b).sum() / max(u, 1)
    sd = ((a & ~b).sum() + (b & ~a).sum()) / max(b.sum(), 1)
    ys, xs = np.nonzero(b); ya, xa = np.nonzero(a)
    cen = float(np.hypot(xs.mean() - xa.mean(), ys.mean() - ya.mean())) / SC if len(xs) and len(xa) else np.nan
    return sd, float(iou), cen


def cost(x, T0, Mk, h):
    O = app(T0, x); sd, _, _ = score(O, Mk, h)
    rot = np.degrees(np.linalg.norm(x[:3] * 0.1)); dz = abs(x[5]) * 1.0
    return sd + 0.08 * (rot / 15) ** 2 + 0.08 * (dz / 3) ** 2 + 0.002 * (x ** 2).sum()


def fit(t):
    Mk, h = load(t)
    if Mk is None: return t, None, None, None, 0
    vis = int((Mk & ~h).sum())
    T0 = CAMold[t]; s0 = score(T0, Mk, h)
    if vis < 150: return t, T0, s0, s0, vis
    starts = [np.zeros(6)]
    a = sil(T0) & ~h; b = Mk & ~h
    if a.sum() and b.sum():
        ya, xa = np.nonzero(a); yb, xb = np.nonzero(b); c = T0[:3, :3] @ COM + T0[:3, 3]; z = c[2]
        dx = (xb.mean() - xa.mean()) / Ks[0, 0] * z; dy = (yb.mean() - ya.mean()) / Ks[1, 1] * z
        starts.append(np.array([0, 0, 0, dx * 100, dy * 100, 0]))
    best = None
    for x0 in starts:
        r = minimize(cost, x0, args=(T0, Mk, h), method='Powell', bounds=[(-5, 5)] * 3 + [(-12, 12)] * 2 + [(-6, 6)], options=dict(maxfev=700, xtol=0.01, ftol=1e-4))
        if best is None or r.fun < best[0]: best = (r.fun, r.x)
    T1 = app(T0, best[1]); return t, T1, s0, score(T1, Mk, h), vis


def inwin(t, W): return any(a * fps <= t <= b * fps for a, b in W)


def ramp_w(N, W, ramp):
    w = np.zeros(N); tt = np.arange(N) / fps
    for a, b in W:
        x = np.clip(np.minimum((tt - a) / ramp + 1, (b - tt) / ramp + 1), 0, 1); w = np.maximum(w, x)
    w[0] = 0; return w


if __name__ == '__main__':
    out = OUTROOT / f'ep{EP:02d}'; out.mkdir(parents=True, exist_ok=True)
    frames = [t for t in range(1, N) if inwin(t, [(a - 0.4, b + 0.4) for a, b in WINS])]
    # lag check on the old target (whole episode, every 4th frame)
    with Pool(WK) as P:
        def_lag = {}
        res = P.map(fit, frames, chunksize=2)
    fitd = {t: (T1, s0, s1, vis) for t, T1, s0, s1, vis in res}
    # corrections in camera frame: COM offset and rotation offset
    dp = np.zeros((N, 3)); dr = np.zeros((N, 3)); wt = np.zeros(N)
    for t, (T1, s0, s1, vis) in fitd.items():
        if T1 is None or s0 is None: continue
        T0 = CAMold[t]; c0 = T0[:3, :3] @ COM + T0[:3, 3]; c1 = T1[:3, :3] @ COM + T1[:3, 3]
        gain = s1[1] - s0[1]
        if vis >= 150 and s1[1] >= 0.25 and gain > -0.02:
            dp[t] = c1 - c0; dr[t] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec(); wt[t] = min(1.0, vis / 600) * (0.5 + s1[1])
    sig = 3.0; k = np.exp(-0.5 * (np.arange(-10, 11) / sig) ** 2)
    num_p = np.stack([np.convolve(dp[:, i] * wt, k, 'same') for i in range(3)], 1); num_r = np.stack([np.convolve(dr[:, i] * wt, k, 'same') for i in range(3)], 1)
    den = np.convolve(wt, k, 'same')
    # fill frames with no support by linear interpolation of supported ones inside the window
    sp = np.where(den > 0.3)[0]
    sdp = np.zeros((N, 3)); sdr = np.zeros((N, 3))
    if len(sp):
        for i in range(3):
            sdp[:, i] = np.interp(np.arange(N), sp, num_p[sp, i] / den[sp]); sdr[:, i] = np.interp(np.arange(N), sp, num_r[sp, i] / den[sp])
    w = ramp_w(N, WINS, 0.25)
    CAMnew = CAMold.copy()
    for t in range(N):
        if w[t] <= 0: continue
        T0 = CAMold[t]; c0 = T0[:3, :3] @ COM + T0[:3, 3]; Q = R.from_rotvec(sdr[t] * w[t]).as_matrix()
        O = T0.copy(); O[:3, :3] = Q @ T0[:3, :3]; O[:3, 3] = Q @ (T0[:3, 3] - c0) + c0 + sdp[t] * w[t]; CAMnew[t] = O
    Wnew = np.einsum('tij,tjk->tik', Wrgb, CAMnew); Tnew = np.einsum('tij,tjk->tik', Sx, Wnew)
    # jump repair in the target frame
    def speeds(T):
        v = np.r_[0, np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 100 * fps]
        a = np.r_[0, np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', T[:-1, :3, :3], T[1:, :3, :3])).magnitude()) * fps]
        return v, a
    rep = []
    if JUMPS:
        v, a = speeds(Tnew); bad0 = np.array([inwin(t, JUMPS) and (v[t] > 100 or (t + 1 < N and v[t + 1] > 100)) for t in range(N)])
        bad = np.zeros(N, bool)
        for ja, jb in JUMPS:   # replace the whole excursion: first..last flagged frame inside each jump window
            idx = [t for t in np.where(bad0)[0] if ja * fps <= t <= jb * fps]
            if idx: bad[min(idx):max(idx) + 1] = True
        bad[0] = False; good = np.where(~bad)[0]
        for t in np.where(bad)[0]:
            lo = good[good < t].max(); hi = good[good > t].min(); s = (t - lo) / (hi - lo)
            sl = Slerp([0, 1], R.from_matrix(np.stack([Tnew[lo, :3, :3], Tnew[hi, :3, :3]])))
            Tnew[t, :3, :3] = sl(s).as_matrix(); Tnew[t, :3, 3] = (1 - s) * Tnew[lo, :3, 3] + s * Tnew[hi, :3, 3]; rep.append(int(t))
        CAMnew = np.einsum('tij,tjk->tik', np.linalg.inv(Wrgb), np.einsum('tij,tjk->tik', np.linalg.inv(Sx), Tnew))
    assert np.allclose(Tnew[0], Told[0])
    # write target
    tg = OUTROOT.parent / "targets" / __import__("os").environ.get("TSET", "tgt_s1") / E
    if tg.exists(): shutil.rmtree(tg)
    shutil.copytree(TGT, tg, symlinks=False); np.savez_compressed(tg / 'wooden_spoon_trajectory.npz', T_world_obj=Tnew)
    # evaluation over all frames in (windows +- 1 s) and the jump windows
    evf = [t for t in range(N) if inwin(t, [(a - 1, b + 1) for a, b in WINS + JUMPS])]
    def ev(t):
        Mk, h = load(t)
        if Mk is None: return t, None, None, 0
        return t, score(CAMold[t], Mk, h), score(CAMnew[t], Mk, h), int((Mk & ~h).sum())
    with Pool(WK) as P: er = P.map(ev, evf, chunksize=4)
    E_ = {t: (a, b, vis) for t, a, b, vis in er}
    vo, ao = speeds(Told); vn, an = speeds(Tnew)
    dpos = np.linalg.norm(Tnew[:, :3, 3] - Told[:, :3, 3], axis=1) * 100
    drot = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', Told[:, :3, :3], Tnew[:, :3, :3])).magnitude())
    rows = []
    t0 = min(evf) / fps; t1 = max(evf) / fps; s = np.floor(t0 * 2) / 2
    while s < t1:
        fr = [t for t in evf if s * fps <= t < (s + 0.5) * fps and E_[t][0] is not None and E_[t][2] >= 150]
        if fr:
            rows.append(dict(t0=s, t1=s + 0.5, n=len(fr), iou_old=float(np.mean([E_[t][0][1] for t in fr])), iou_new=float(np.mean([E_[t][1][1] for t in fr])),
                             cen_old_px=float(np.nanmean([E_[t][0][2] for t in fr])), cen_new_px=float(np.nanmean([E_[t][1][2] for t in fr])),
                             dpos_cm=float(dpos[fr].mean()), drot_deg=float(drot[fr].mean()), vmax_old=float(vo[fr].max()), vmax_new=float(vn[fr].max())))
        s += 0.5
    ch = w > 0
    summ = dict(ep=EP, windows=WINS, jumps=JUMPS, repaired_frames=rep, frames_changed=int((dpos > 1e-6).sum()),
                mean_dpos_cm_changed=float(dpos[dpos > 1e-6].mean()) if (dpos > 1e-6).any() else 0, mean_drot_deg_changed=float(drot[dpos > 1e-6].mean()) if (dpos > 1e-6).any() else 0,
                max_speed_old=[float(vo[1:].max()), float(ao[1:].max())], max_speed_new=[float(vn[1:].max()), float(an[1:].max())],
                iou_in_windows=[float(np.mean([r['iou_old'] for r in rows if any(a <= r['t0'] < b for a, b in WINS)] or [np.nan])), float(np.mean([r['iou_new'] for r in rows if any(a <= r['t0'] < b for a, b in WINS)] or [np.nan]))],
                f0_unchanged=True, target=str(tg), source_target=str(TGT))
    (out / 'report.json').write_text(json.dumps(dict(summary=summ, windows=rows), indent=1))
    np.savez(out / 'fit.npz', CAMold=CAMold, CAMnew=CAMnew, Told=Told, Tnew=Tnew, w=w, wt=wt)
    print(json.dumps(summ))
    for r in rows: print('%5.2f-%5.2f n%2d IoU %.2f -> %.2f  cen %5.1f -> %5.1f px  d %.1f cm %.1f deg  vmax %.0f -> %.0f' % (r['t0'], r['t1'], r['n'], r['iou_old'], r['iou_new'], r['cen_old_px'], r['cen_new_px'], r['dpos_cm'], r['drot_deg'], r['vmax_old'], r['vmax_new']))
    # video: RGB | old | new over evf frames
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe(); vs = 0.6; pw, ph = int(aw * vs), int(ah * vs)
    vid = out / f'ep{EP:02d}_spoon_recon_old_vs_new.mp4'
    wr = subprocess.Popen([ff, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{pw*3}x{ph+40}', '-r', str(fps), '-i', '-', '-an',
                           '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p', str(vid)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(man['source_video']); t = 0; lo, hi = min(evf), max(evf)
    while t <= hi:
        ok, fr = cap.read()
        if not ok: break
        if t >= lo:
            sm = cv2.resize(fr, (aw, ah)); Mk, h = load(t); panes = [sm.copy()]
            for T, col in ((CAMold[t], (0, 120, 255)), (CAMnew[t], (60, 220, 60))):
                p = sm.copy(); S = sil(T); p[S] = (0.45 * p[S] + 0.55 * np.array(col)).astype(np.uint8)
                if Mk is not None:
                    cs, _ = cv2.findContours(Mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(p, cs, -1, (255, 255, 255), 2)
                panes.append(p)
            row = np.hstack([cv2.resize(p, (pw, ph)) for p in panes]); bar = np.full((40, row.shape[1], 3), 24, np.uint8)
            e = E_.get(t); txt = f'ep{EP} t={t/fps:5.2f}s  RGB | old target (orange) + SAM2 outline' + (f' IoU {e[0][1]:.2f}' if e and e[0] else '') + ' | new target (green)' + (f' IoU {e[1][1]:.2f}' if e and e[1] else '') + ('  [in window]' if w[t] > 0 else '')
            cv2.putText(bar, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1, cv2.LINE_AA); wr.stdin.write(np.vstack([bar, row]).tobytes())
        t += 1
    wr.stdin.close(); wr.wait(); print('VIDEO', vid)
