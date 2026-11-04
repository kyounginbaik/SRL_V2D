"""ep3 STEREO-ONLY spoon refit (canonical camera: canonical masks with hands + pot as occluders, stereo depth, marker balls; no RGB/SAM2,
no GT, no human marks). Windows 5.5-7.0, 8.0-8.5, 10.5-11.0 s. Per frame: 6-DoF correction about the COM (Powell) of the FINAL pose;
accepted only if BOTH the ball error and the stereo IoU improve. Accepted corrections are smoothed (sigma 3 frames, weighted) and ramped
(0.25 s) at the window edges; f0 unchanged. The pot is left unchanged. Writes TMP then renames to tgt_e3s/episode_000003."""
import sys, json, shutil, os, numpy as np, cv2, trimesh
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA
from guard_clip import Evidence, CAN
from retime import shift
EP = 3; E = 'episode_000003'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
BUN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
TG = C22 / 'streamALL20K/targets/tgt_v32' / E; WINS = [(5.5, 7.0), (8.0, 8.5), (10.5, 11.0)]; fps = 20.0
z = np.load(BUN / 'object_poses.npz'); bn = [str(n) for n in z['names']]; si = bn.index('wooden_spoon'); F = len(z['camera_from_object'])
root = CAN / 'evaluation' / E; dd = DELTA[f'evaluation_{EP}']; dd = np.asarray(dd, float)[:F] if np.ndim(dd) else np.full(F, float(dd))
I = json.loads((root / 'intrinsics.json').read_text()); fx, fy, cx, cy, W, H = I['fx'], I['fy'], I['cx'], I['cy'], I['width'], I['height']
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
mj = json.loads((BUN / 'meshes.json').read_text()); mesh = BUN / mj['wooden_spoon']['file']
m = trimesh.load(mesh, force='mesh', process=False); V = np.asarray(m.vertices); Fc = np.asarray(m.faces); COM = V.mean(0)
fcn = V[Fc].mean(1); fn = np.asarray(m.face_normals); rs = np.random.default_rng(0).choice(len(Fc), min(800, len(Fc)), replace=False); fcn, fn = fcn[rs], fn[rs]
B = np.asarray(json.loads((root / 'ball_models_v1.json').read_text())['wooden_spoon']); DET = json.loads((root / 'ball_detections_wooden_spoon.json').read_text())['frames']
camS = z['camera_from_object'][:, si].astype(np.float64); camC = shift(camS, -dd)   # RGB/canonical-image clock, as spoon_annot_roll3
SC = 0.5; aw, ah = int(W * SC), int(H * SC)


def rd(p): return cv2.imread(str(p), -1)


def load(i):
    o = root / 'objects'
    try:
        Mk = rd(o / f'wooden_spoon/masks/1/{i:06d}.png') > 0
        h = (rd(o / f'wooden_spoon/masks/2/{i:06d}.png') > 0) | (rd(o / f'wooden_spoon/masks/3/{i:06d}.png') > 0)
        pp = o / f'white_pot/masks/1/{i:06d}.png'
        if pp.exists(): h = h | (rd(pp) > 0)
    except Exception:
        return None
    dv = rd(root / f'depth/{i:06d}.png')
    D = None if dv is None else np.where(dv < 65535, 65535 / np.maximum(dv.astype(np.float64), 1) - 1, np.nan)
    r = lambda x: cv2.resize(x.astype(np.uint8), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
    return r(Mk), r(h), D, (DET[i] if i < len(DET) else [])


def sil(T):
    p = V @ T[:3, :3].T + T[:3, 3]
    if (p[:, 2] < 0.02).any(): return np.zeros((ah, aw), bool)
    u = p[:, :2] / p[:, 2:3] @ K[:2, :2].T + K[:2, 2]; img = np.zeros((ah, aw), np.uint8)
    cv2.fillPoly(img, list(np.round(u[Fc] * SC * 4).astype(np.int32)), 1, shift=2); return img > 0


def parts(T, Mk, h, D, det):
    S = sil(T); a = S & ~h; b = Mk & ~h; iou = (a & b).sum() / max((a | b).sum(), 1); sd = ((b & ~a).sum() + (a & ~b).sum()) / max(b.sum(), 1)
    dz = np.nan
    if D is not None:
        pc = fcn @ T[:3, :3].T + T[:3, 3]; nc = fn @ T[:3, :3].T; vis = (nc * pc).sum(1) < 0
        u = (pc[vis, :2] / pc[vis, 2:3] @ K[:2, :2].T + K[:2, 2]).astype(int); ok = (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H); u = u[ok]; zz = pc[vis][ok, 2]
        if len(u) > 20:
            Dv = D[u[:, 1], u[:, 0]]; us = (u * SC).astype(int); us[:, 0] = np.clip(us[:, 0], 0, aw - 1); us[:, 1] = np.clip(us[:, 1], 0, ah - 1)
            keep = np.isfinite(Dv) & Mk[us[:, 1], us[:, 0]] & ~h[us[:, 1], us[:, 0]]
            if keep.sum() > 10: dz = float(np.median(np.abs(zz[keep] - Dv[keep])))
    bl = np.nan
    if det:
        c = T[:3, :3] @ COM + T[:3, 3]; dl = [d for d in det if 'camera_point' not in d or np.linalg.norm(np.asarray(d['camera_point']) - c) < 0.15]
        if dl:
            ub = (B @ T[:3, :3].T + T[:3, 3]); ub = ub[:, :2] / ub[:, 2:3] @ K[:2, :2].T + K[:2, 2]
            bl = float(np.median([min(30, np.min(np.linalg.norm(ub - np.asarray(d['uv']), axis=1))) for d in dl]))
    return sd, float(iou), dz, bl


def app(T, x):
    O = T.copy(); Q = R.from_rotvec(x[:3] * 0.1).as_matrix(); c = T[:3, :3] @ COM + T[:3, 3]
    O[:3, :3] = Q @ T[:3, :3]; O[:3, 3] = Q @ (T[:3, 3] - c) + c + x[3:] * 0.01; return O


def cost(x, T0, dat):
    sd, iou, dz, bl = parts(app(T0, x), *dat); c = sd + (0 if np.isnan(dz) else min(dz, .05) / .02 * .15) + 0.002 * (x ** 2).sum()
    if np.isfinite(bl): c += bl / 30 * 1.0
    return c


def fit(f):
    dat = load(f)
    if dat is None: return f, None, None, None
    T0 = camC[f]; p0 = parts(T0, *dat)
    if (dat[0] & ~dat[1]).sum() < 150: return f, None, p0, p0
    r = minimize(cost, np.zeros(6), args=(T0, dat), method='Powell', bounds=[(-6, 6)] * 3 + [(-5, 5)] * 3, options=dict(maxfev=600, xtol=0.01, ftol=1e-4))
    T1 = app(T0, r.x); return f, T1, p0, parts(T1, *dat)


def inwin(f, Ws): return any(a * fps <= f <= b * fps for a, b in Ws)


if __name__ == '__main__':
    fr = [f for f in range(1, F) if inwin(f, [(a - 0.3, b + 0.3) for a, b in WINS])]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as P: res = P.map(fit, fr, chunksize=2)
    dp = np.zeros((F, 3)); dr = np.zeros((F, 3)); wt = np.zeros(F); acc = 0; tried = 0
    for f, T1, p0, p1 in res:
        if T1 is None: continue
        tried += 1
        bo, bnn = p0[3], p1[3]
        if not (np.isfinite(bo) and np.isfinite(bnn) and bnn < bo - 0.5 and p1[1] > p0[1] + 0.005): continue
        acc += 1; T0 = camC[f]; dp[f] = (T1[:3, :3] @ COM + T1[:3, 3]) - (T0[:3, :3] @ COM + T0[:3, 3]); dr[f] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec(); wt[f] = 1.0
    k = np.exp(-0.5 * (np.arange(-10, 11) / 3.0) ** 2); den = np.convolve(wt, k, 'same'); sp = np.where(den > 0.3)[0]
    sdp = np.zeros((F, 3)); sdr = np.zeros((F, 3))
    if len(sp):
        for i in range(3):
            sdp[:, i] = np.interp(np.arange(F), sp, np.convolve(dp[:, i] * wt, k, 'same')[sp] / den[sp], left=0, right=0)
            sdr[:, i] = np.interp(np.arange(F), sp, np.convolve(dr[:, i] * wt, k, 'same')[sp] / den[sp], left=0, right=0)
    tt = np.arange(F) / fps; w = np.zeros(F)
    for a, b in WINS: w = np.maximum(w, np.clip(np.minimum((tt - a) / 0.25 + 1, (b - tt) / 0.25 + 1), 0, 1))
    w[0] = 0
    # apply the smoothed camera-frame correction to the STORED-clock camera pose (corrections are smooth; ~0.9 frame clock offset ignored)
    camN = camS.copy()
    for g in np.where(w > 0)[0]:
        T0 = camS[g]; c0 = T0[:3, :3] @ COM + T0[:3, 3]; Q = R.from_rotvec(sdr[g] * w[g]).as_matrix()
        camN[g, :3, :3] = Q @ T0[:3, :3]; camN[g, :3, 3] = Q @ (T0[:3, 3] - c0) + c0 + sdp[g] * w[g]
    Wold = z['world_from_object'][:, si].astype(np.float64); wfc = np.einsum('tij,tjk->tik', Wold, np.linalg.inv(camS)); Wnew = np.einsum('tij,tjk->tik', wfc, camN)
    Told = np.load(TG / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); assert np.abs(Told - Wold).max() < 1e-9
    Tnew = Wnew; assert np.allclose(Tnew[0], Told[0])
    # evaluation (canonical Evidence IoU + balls), old vs new, in the canonical-image clock
    ev = Evidence(root, ['wooden_spoon'], {'wooden_spoon': mesh}, F); cN = shift(camN, -dd)
    fe = [f for f in range(int(4.5 * fps), int(12 * fps))]
    io = ev.eval('wooden_spoon', camC, fe)[0]; inn = ev.eval('wooden_spoon', cN, fe)[0]
    def bpx(T, f):
        d = DET[f] if f < len(DET) else []; c = T[:3, :3] @ COM + T[:3, 3]; d = [x for x in d if 'camera_point' not in x or np.linalg.norm(np.asarray(x['camera_point']) - c) < 0.15]
        if not d: return np.nan
        ub = B @ T[:3, :3].T + T[:3, 3]; ub = ub[:, :2] / ub[:, 2:3] @ K[:2, :2].T + K[:2, 2]
        return float(np.median([min(30, np.min(np.linalg.norm(ub - np.asarray(x['uv']), axis=1))) for x in d]))
    bo = np.array([bpx(camC[f], f) for f in range(F)]); bnw = np.array([bpx(cN[f], f) for f in range(F)])
    dpos = np.linalg.norm(Tnew[:, :3, 3] - Told[:, :3, 3], axis=1) * 100; drot = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', Told[:, :3, :3], Tnew[:, :3, :3])).magnitude())
    v = lambda T: np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 100 * fps
    rows = []
    for a in np.arange(4.5, 12, 0.5):
        f = [x for x in fe if a * fps <= x < (a + 0.5) * fps]
        rows.append(dict(t0=float(a), iou_old=float(np.nanmean(io[f])), iou_new=float(np.nanmean(inn[f])),
                         ball_old=float(np.nanmedian(bo[f])) if np.isfinite(bo[f]).any() else None, ball_new=float(np.nanmedian(bnw[f])) if np.isfinite(bnw[f]).any() else None,
                         dpos_cm=float(dpos[f].mean()), drot_deg=float(drot[f].mean())))
    ch = np.where(dpos > 1e-6)[0]
    summ = dict(frames_tried=tried, frames_accepted=acc, changed_frames=[int(ch.min()), int(ch.max()), int(len(ch))] if len(ch) else [],
                changed_seconds=[[round(a - 0.25, 2), round(b + 0.25, 2)] for a, b in WINS] if len(ch) else [], max_dpos_cm=float(dpos.max()), max_drot_deg=float(drot.max()),
                max_speed_old=float(v(Told).max()), max_speed_new=float(v(Tnew).max()),
                ball_median_windows=[float(np.nanmedian(bo[[f for f in fe if inwin(f, WINS)]])), float(np.nanmedian(bnw[[f for f in fe if inwin(f, WINS)]]))],
                iou_windows=[float(np.nanmean(io[[f - fe[0] for f in fe if inwin(f, WINS)]])) if False else float(np.nanmean([io[f] for f in fe if inwin(f, WINS)])), float(np.nanmean([inn[f] for f in fe if inwin(f, WINS)]))],
                pot='unchanged (stereo IoU 0.69-0.81, ball-to-surface 0.0-0.5 cm in every window: no tilt evidence)')
    OUT = Path('/data1/hansol/spoon3537/targets/tgt_e3s') / E; TMP = Path('/data1/hansol/spoon3537/targets/.tmp_e3s') / E
    if TMP.parent.exists(): shutil.rmtree(TMP.parent)
    shutil.copytree(TG, TMP, symlinks=False)
    if len(ch): np.savez_compressed(TMP / 'wooden_spoon_trajectory.npz', T_world_obj=Tnew)
    (Path('/data1/hansol/spoon3537/work/e3s')).mkdir(parents=True, exist_ok=True)
    Path('/data1/hansol/spoon3537/work/e3s/report.json').write_text(json.dumps(dict(summary=summ, windows=rows), indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): shutil.rmtree(OUT)
    os.rename(TMP, OUT); shutil.rmtree(TMP.parent)
    print(json.dumps(summ))
    for r in rows: print('%5.1f IoU %.2f -> %.2f  ball %s -> %s  d %.2f cm %.2f deg' % (r['t0'], r['iou_old'], r['iou_new'], '%.1f' % r['ball_old'] if r['ball_old'] is not None else ' nan', '%.1f' % r['ball_new'] if r['ball_new'] is not None else ' nan', r['dpos_cm'], r['drot_deg']))
    print('TARGET', OUT, __import__('hashlib').sha256((OUT / 'wooden_spoon_trajectory.npz').read_bytes()).hexdigest())
