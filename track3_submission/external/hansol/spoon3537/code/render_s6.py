"""ep36/37/35 spoon recon check video with bowl front/back: RGB | old target | tgt_s1 | tgt_s2 (if present).
Each target's spoon is drawn face by face (far to near). Bowl inner (concave) faces = blue, bowl outer (convex) faces = red,
handle = grey. A yellow arrow shows the concave normal from the bowl centre. White = SAM2 spoon outline.
The label gives the facing (+1: concave side to the camera, -1: back to the camera). GT-free, render only.
usage: render_ff.py EP OUT.mp4 T0 T1"""
import sys, json, subprocess, numpy as np, cv2, trimesh
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from spoon_hand_bowl import bowl_normal

EP = int(sys.argv[1]); OUT = Path(sys.argv[2]); T0, T1 = float(sys.argv[3]), float(sys.argv[4])
E = f'episode_{EP:06d}'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
COMP = {35: 'streamK35_ep35/staging/K35_H2joint_b1/components/episode_000035', 36: 'streamALL20K/staging/tgt_v32_ep36_b2/components/episode_000036',
        37: 'streamALL20K/staging/tgt_v32_ep37_b1/components/episode_000037'}[EP]
TGT = {35: C22 / 'streamK35_ep35/targets/K35_H2joint/episode_000035', 36: C22 / 'target_final_v14_g17/episode_000036', 37: C22 / 'target_final_v14_g17/episode_000037'}[EP]
man = json.loads((C22 / COMP / f'portable_validation/runtime/episodes/{E}/input_manifest.json').read_text()); SRC = Path(man['source'])
SAMO = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs') / E / 'wooden_spoon'
op = np.load(SRC / 'object_poses.npz'); si = [str(n) for n in op['names']].index('wooden_spoon')
Wold = op['world_from_object'][:, si].astype(np.float64); Wrgb = op['world_from_rgb'].astype(np.float64)
Told = np.load(TGT / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); N = len(Told)
Sxi = np.linalg.inv(np.einsum('tij,tjk->tik', Told, np.linalg.inv(Wold)))
cam = lambda T: np.einsum('tij,tjk->tik', np.linalg.inv(Wrgb), np.einsum('tij,tjk->tik', Sxi, T))
SETS = [("FINAL", cam(Told))]
for s in ("tgt_s6",):
    p = Path('/data1/hansol/spoon3537/targets') / s / E / 'wooden_spoon_trajectory.npz'
    if p.exists(): SETS.append((s, cam(np.load(p)['T_world_obj'].astype(np.float64))))
fps = float(man['fps'])
mp = SRC / 'assets/wooden_spoon.obj'; m = trimesh.load(mp, force='mesh', process=False)
V = np.asarray(m.vertices); F = np.asarray(m.faces); nb = bowl_normal(str(mp))[0]
c = V.mean(0); w, U = np.linalg.eigh(np.cov((V - c).T)); ax = U[:, 2]; s = (V - c) @ ax; lo, hi = s.min(), s.max(); L = hi - lo
wid = lambda msk: np.linalg.norm((V[msk] - c) - np.outer(s[msk], ax), axis=1).max()
if wid(s < lo + 0.3 * L) > wid(s > hi - 0.3 * L): ax, s = -ax, -s; lo, hi = s.min(), s.max()
bowlv = s > hi - 0.38 * L; bc = V[s > hi - 0.3 * L].mean(0)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
fbowl = bowlv[F].mean(1) > 0.5; fdot = fn @ nb
FCOL = np.tile(np.array([150, 150, 150], float), (len(F), 1))
FCOL[fbowl & (fdot > 0.2)] = (235, 120, 30)      # BGR blue-ish: concave
FCOL[fbowl & (fdot < -0.2)] = (40, 40, 220)      # BGR red: convex back
d = json.loads((SAMO / 'anycalib/undistorted_intrinsics.json').read_text()); SC = 0.5
K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1.]]); WH = (int(d['width']), int(d['height']))
aw, ah = int(WH[0] * SC), int(WH[1] * SC); Ks = K.copy(); Ks[:2] *= SC
proj = lambda P: P[:, :2] / P[:, 2:3] @ Ks[:2, :2].T + Ks[:2, 2]


def draw(img, T):
    P = V @ T[:3, :3].T + T[:3, 3]
    if (P[:, 2] < 0.02).any(): return None
    uv = proj(P); fz = P[F].mean(1)[:, 2]; nc = fn @ T[:3, :3].T; vis = (nc * P[F].mean(1)).sum(1) < 0
    ov = img.copy(); light = 0.55 + 0.45 * np.clip(-(nc * (P[F].mean(1) / np.linalg.norm(P[F].mean(1), axis=1, keepdims=True))).sum(1), 0, 1)
    for i in np.argsort(-fz):
        if not vis[i]: continue
        cv2.fillConvexPoly(ov, np.round(uv[F[i]] * 4).astype(np.int32), tuple(float(x) for x in FCOL[i] * light[i]), lineType=cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(ov, 0.75, img, 0.25, 0)
    b = T[:3, :3] @ bc + T[:3, 3]; n = T[:3, :3] @ nb; a0, a1 = proj(np.stack([b, b + 0.05 * n]))
    cv2.arrowedLine(out, tuple(np.round(a0).astype(int)), tuple(np.round(a1).astype(int)), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.3)
    return out, float(n @ (-b / np.linalg.norm(b)))


frames = [t for t in range(N) if T0 * fps <= t <= T1 * fps]


def one(t):
    cap = cv2.VideoCapture(man['source_video']); cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, fr = cap.read()
    if not ok: return t, None
    sm = cv2.resize(fr, (aw, ah)); mp_ = SAMO / f'masks/1/{t:06d}.png'
    cs = []
    if mp_.exists():
        Mk = cv2.resize(cv2.imread(str(mp_), cv2.IMREAD_GRAYSCALE), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
        cs, _ = cv2.findContours(Mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panes = [sm.copy()]; labs = ['RGB']
    for name, C in SETS:
        r = draw(sm.copy(), C[t])
        p, fc = (sm.copy(), float('nan')) if r is None else r
        cv2.drawContours(p, cs, -1, (255, 255, 255), 1); panes.append(p); labs.append(f'{name}  facing {fc:+.2f}')
    vs = 0.9; pw, ph = int(aw * vs), int(ah * vs)
    row = np.hstack([cv2.resize(p, (pw, ph)) for p in panes]); bar = np.full((36, row.shape[1], 3), 24, np.uint8)
    for k, lb in enumerate(labs):
        cv2.putText(bar, (f'ep{EP} t={t / fps:5.2f}s  ' if k == 0 else '') + lb, (k * pw + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
    return t, np.vstack([bar, row])


if __name__ == '__main__':
    import imageio_ffmpeg
    with Pool(8) as P: res = dict(P.map(one, frames, chunksize=4))
    fr0 = next(v for v in res.values() if v is not None); Hh, Ww = fr0.shape[:2]
    wr = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{Ww}x{Hh}', '-r', str(fps), '-i', '-', '-an',
                           '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', str(OUT)], stdin=subprocess.PIPE)
    for t in frames:
        if res[t] is not None: wr.stdin.write(res[t].tobytes())
    wr.stdin.close(); wr.wait()
    for ts in (14.5, 15.0, 15.15, 15.5):
        t = int(round(ts * fps))
        if t in res and res[t] is not None: cv2.imwrite(str(OUT.with_name(OUT.stem + f'_t{ts:.2f}.jpg')), res[t], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('VIDEO', OUT, len(frames), 'frames; sets', [s for s, _ in SETS])
