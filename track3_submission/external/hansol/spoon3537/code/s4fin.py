"""ep36 tgt_s4 finalize: apply the roll fit (WORK/roll/bundle) on top of T4a, write tgt_s4, and evaluate vs tgt_s1:
IoU / centre error per 0.5 s (11-18 s, SAM2 masks, hands excluded), max speeds, marker-ball median (canonical camera, frames 250-350),
bowl concave-normal image direction at 15.0-15.5 s. usage: s4fin.py WORK"""
import sys, json, shutil, numpy as np, cv2
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
sys.argv = [sys.argv[0], sys.argv[1]]
WORK = Path(sys.argv[1])
exec(open('/data1/hansol/spoon3537/code/s4pos.py').read().split("if __name__ == '__main__':")[0].replace("WORK = Path(sys.argv[1]); WORK.mkdir(parents=True, exist_ok=True)", ""))
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import CleanBalls, DELTA
from ep32_late_roll import ball_px
from guard_clip import CAN
from retime import shift
from spoon_hand_bowl import bowl_normal

T4a = np.load(WORK / 'T4a.npz')['T4a']
zb = np.load(WORK / 'bundle' / E / 'object_poses.npz'); zr = np.load(WORK / 'roll/bundle' / E / 'object_poses.npz')
s = [str(n) for n in zb['names']].index('wooden_spoon')
Rr = np.einsum('tij,tjk->tik', np.linalg.inv(zb['world_from_object'][:, s]), zr['world_from_object'][:, s])
T4 = np.einsum('tij,tjk->tik', T4a, Rr); assert np.allclose(T4[0], Told[0])
S4 = Path('/data1/hansol/spoon3537/targets/tgt_s4') / E
if S4.exists(): shutil.rmtree(S4)
shutil.copytree(S1, S4); np.savez_compressed(S4 / 'wooden_spoon_trajectory.npz', T_world_obj=T4)
SETS = dict(s1=T1, s4a=T4a, s4=T4)
CAM = {k: tocam(v) for k, v in SETS.items()}


def ev(t):
    Mk, h = load(t)
    if Mk is None or (Mk & ~h).sum() < 150: return t, None
    return t, {k: score(CAM[k][t], Mk, h) for k in ('s1', 's4')}


if __name__ == '__main__':
    fr = [t for t in range(N) if 11 * fps <= t < 18 * fps]
    with Pool(8) as P: E_ = dict(P.map(ev, fr, chunksize=4))
    rows = []
    for a in np.arange(11, 18, 0.5):
        f = [t for t in fr if a * fps <= t < (a + 0.5) * fps and E_[t] is not None]
        if f: rows.append(dict(t0=float(a), n=len(f), iou_s1=float(np.mean([E_[t]['s1'][1] for t in f])), iou_s4=float(np.mean([E_[t]['s4'][1] for t in f])),
                               cen_s1=float(np.nanmean([E_[t]['s1'][2] for t in f])), cen_s4=float(np.nanmean([E_[t]['s4'][2] for t in f])),
                               area_s4_over_mask=float(np.mean([E_[t]['s4'][3] / max(1, (load(t)[0] & ~load(t)[1]).sum()) for t in f]))))
    def speeds(T):
        v = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 100 * fps
        a = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', T[:-1, :3, :3], T[1:, :3, :3])).magnitude()) * fps
        sel = slice(int(12 * fps), int(18 * fps)); return float(v[sel].max()), float(a[sel].max()), float(v.max()), float(a.max())
    # marker balls in the canonical camera (bundle frame, stored->RGB clock shift as in spoon_annot_roll3)
    root = CAN / 'evaluation' / E; Fz = len(zb['camera_from_object']); dd = DELTA['evaluation_36']; dd = np.asarray(dd, float)[:Fz] if np.ndim(dd) else np.full(Fz, float(dd))
    I = json.loads((root / 'intrinsics.json').read_text()); Kc = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]]); balls = CleanBalls(root, Fz)
    zf = np.load(BUN / 'object_poses.npz'); camF = zf['camera_from_object'][:, s]
    bm = {}
    for k, T in SETS.items():
        cb = shift(np.einsum('tij,tjk->tik', camF, np.einsum('tij,tjk->tik', np.linalg.inv(Told), T)), -dd)
        x = np.array([ball_px(balls.model, cb[f], Kc, balls.det[f]) if balls.ok and f < len(balls.det) else np.nan for f in range(250, 351)])
        bm[k] = dict(median=float(np.nanmedian(x)), frames=int(np.isfinite(x).sum()), median_14p5_16p5=float(np.nanmedian(x[290 - 250:331 - 250])))
    # concave normal image direction at 15.0-15.5 s (x right, y down; down = toward the table in this view)
    nb = bowl_normal(str(SRC / 'assets/wooden_spoon.obj'))[0]; c = V.mean(0); _, U = np.linalg.eigh(np.cov((V - c).T)); ax = U[:, 2]; sp = (V - c) @ ax
    lo, hi = sp.min(), sp.max(); L = hi - lo
    wid = lambda msk: np.linalg.norm((V[msk] - c) - np.outer(sp[msk], ax), axis=1).max()
    if wid(sp < lo + 0.3 * L) > wid(sp > hi - 0.3 * L): ax, sp = -ax, -sp; lo, hi = sp.min(), sp.max()
    bc = V[sp > hi - 0.3 * L].mean(0)
    def arrow(C, t):
        b = C[t, :3, :3] @ bc + C[t, :3, 3]; n = C[t, :3, :3] @ nb
        p0 = b[:2] / b[2] * [Ks[0, 0], Ks[1, 1]]; p1 = (b + 0.05 * n)[:2] / (b + 0.05 * n)[2] * [Ks[0, 0], Ks[1, 1]]; v = p1 - p0
        return [round(float(x), 2) for x in v / (np.linalg.norm(v) + 1e-9)] + [round(float(n @ (-b / np.linalg.norm(b))), 2)]
    arr = {f'{t / fps:.2f}': dict(s1=arrow(CAM['s1'], t), s4=arrow(CAM['s4'], t)) for t in range(296, 316, 3)}
    zc = lambda C, t: round(100 * float((C[t, :3, :3] @ c + C[t, :3, 3])[2]), 1)
    out = dict(target=str(S4), sha=__import__('hashlib').sha256((S4 / 'wooden_spoon_trajectory.npz').read_bytes()).hexdigest(),
               speeds_12_18_and_all=dict((k, speeds(v)) for k, v in SETS.items()), ball=bm, arrow_dxdy_facing=arr,
               cam_z_cm_15p15=dict((k, zc(CAM[k], 303)) for k in SETS), roll_change_deg_max=float(np.degrees(R.from_matrix(Rr[:, :3, :3]).magnitude()).max()), windows=rows)
    (WORK / 's4_report.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'windows'}))
    for r in rows: print('%5.1f n%2d IoU s1 %.2f s4 %.2f | cen s1 %5.1f s4 %5.1f px | area s4/mask %.2f' % (r['t0'], r['n'], r['iou_s1'], r['iou_s4'], r['cen_s1'], r['cen_s4'], r['area_s4_over_mask']))
