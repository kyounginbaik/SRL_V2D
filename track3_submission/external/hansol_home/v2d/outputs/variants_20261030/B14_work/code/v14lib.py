"""Variant B on the v14_g17 / g15_camrt clock. The shipped B1/B2 rules (B_work/rules.json, B_work/code/variants.py) are re-applied
with every family track first put through the exact G15 operator the team used for the base (streamG15 build_bundle.py /
mk_cam_export.py + retime.py): world = shift(G15camera @ camera_from_object, delta), camera = shift(camera_from_object, delta).
Checked: this operator on v8_g10b / r8_base reproduces v14_g17 / g15_camrt exactly (code/chk_g15.py).
Family frames that are missing / non-finite are filled with the pre-G15 base before the shift (B1) or get weight 0 (B2,
validity mask carried through the shift). GT is not read here."""
import sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/hansol/v2d/outputs/variants_20261030/B_work/code'); import variants as V
G15 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15')
sys.path.insert(0, str(G15 / 'code')); from retime import shift
DELTA = json.loads((G15 / 'out/sync_delta_t.json').read_text())
RULES_B = json.load(open('/home/hansol/v2d/outputs/variants_20261030/B_work/rules.json'))
SHIPPED = (RULES_B['B1'], RULES_B['B2_good'])


def idx(F, d):
    t = np.clip(np.arange(F) + np.asarray(d, float)[:F] if np.ndim(d) else np.arange(F) + d, 0, F - 1)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, F - 1); return i0, i1, t - i0


def family_cam(exports_dir, fam, name, F):
    p = os.path.join(exports_dir, fam, 'object_poses.npz')
    if not os.path.exists(p): return None
    a = np.load(p); an = [str(x) for x in a['names']]
    if name not in an: return None
    C = np.full((F, 4, 4), np.nan); Ca = a['camera_from_object'][:, an.index(name)]; Fa = min(F, len(Ca)); C[:Fa] = Ca[:Fa]; return C


def g15(C_pre, ok, cam, d, retime=True):
    """C_pre: family camera_from_object on the RGB clock, already filled with the pre-G15 base where not ok.
    Returns world, camera on the v14 clock (or, retime=False, the naive version that keeps the RGB clock) and the shifted mask."""
    Wp = cam @ C_pre
    if not retime: return Wp, C_pre.copy(), ok.copy()
    i0, i1, a = idx(len(C_pre), d)
    return shift(Wp, d), shift(C_pre, d), ok[i0] & (ok[i1] | (a < 1e-9))


def consensus2(bW, bC, cW, cC, oks, name, mesh_path):
    """V.consensus with the weights/twins computed once (world frame, vs the base) and applied to both world and camera poses."""
    F = len(bW); Hs = V.principal_twins(mesh_path) if name in V.TWIN_TYPES else [np.eye(4)]
    outs = []
    Wts = [np.ones(F)]; Pw = [bW[:, :3, 3]]; Pc = [bC[:, :3, 3]]
    from scipy.spatial.transform import Rotation as Rot
    Qw = [Rot.from_matrix(bW[:, :3, :3]).as_quat()]; Qc = [Rot.from_matrix(bC[:, :3, :3]).as_quat()]
    info = dict(candidates=len(cW), mean_weight=[], twin_frames=[])
    for Cw, Cc, ok in zip(cW, cC, oks):
        best = None
        for h, H in enumerate(Hs):
            Ch = Cw @ H; dp, dr = V.dist(Ch, bW); d = np.maximum(dp / V.SIG_P, (0 if name in V.SYMMETRIC else dr) / V.SIG_R)
            best = (np.full(F, h), d) if best is None else (np.where(d < best[1], h, best[0]), np.minimum(d, best[1]))
        hh, d = best; w = np.exp(-0.5 * d ** 2) * ok
        Hf = np.stack(Hs)[hh]; Aw = Cw @ Hf; Ac = Cc @ Hf
        qw = Rot.from_matrix(Aw[:, :3, :3]).as_quat(); qw *= np.sign((qw * Qw[0]).sum(1, keepdims=True) + 1e-12)
        qc = Rot.from_matrix(Ac[:, :3, :3]).as_quat(); qc *= np.sign((qc * Qc[0]).sum(1, keepdims=True) + 1e-12)
        Wts.append(w); Pw.append(Aw[:, :3, 3]); Pc.append(Ac[:, :3, 3]); Qw.append(qw); Qc.append(qc)
        info['mean_weight'].append(float(w.mean())); info['twin_frames'].append(int(((hh > 0) & (w > 0.01)).sum()))
    Wt = np.stack(Wts)
    res = []
    for base, P, Q in [(bW, Pw, Qw), (bC, Pc, Qc)]:
        out = base.copy(); out[:, :3, 3] = (np.stack(P) * Wt[..., None]).sum(0) / Wt.sum(0)[:, None]
        if name not in V.SYMMETRIC:
            qm = (np.stack(Q) * Wt[..., None]).sum(0); qm /= np.linalg.norm(qm, axis=1, keepdims=True); out[:, :3, :3] = Rot.from_quat(qm).as_matrix()
        res.append(out)
    return res[0], res[1], info


def apply14(base14, C_pre_base, exports_dir, cam, d, meshes, rules, variant, retime=True):
    """base14: dict of the v14-clock npz (v14_g17 eval / g15_camrt public); C_pre_base: pre-G15 camera_from_object of the base
    (v8_g10b / r8_base; checked to map exactly onto base14 through g15()). Returns new world, camera arrays and info."""
    b1, good = rules; names = [str(x) for x in base14['names']]; F = len(base14['world_from_object'])
    T = base14['world_from_object'].copy(); Cm = base14['camera_from_object'].copy(); info = {}
    for k, n in enumerate(names):
        fams = [b1[n]] if variant == 'B1' and n in b1 else (good.get(n, []) if variant == 'B2' else [])
        cands = []
        for f in fams:
            C = family_cam(exports_dir, f, n, F)
            if C is None: continue
            ok = np.isfinite(C).all((1, 2)); Cf = np.where(ok[:, None, None], C, C_pre_base[:, k]); cands.append((f, *g15(Cf, ok, cam, d, retime)))
        if not cands:
            if fams: info[n] = dict(source='base (family missing)')
            continue
        if variant == 'B1':
            f, Wf, Cf, ok = cands[0]; T[:, k] = Wf; Cm[:, k] = Cf; info[n] = dict(source=f, coverage=float(ok.mean()))
        else:
            T[:, k], Cm[:, k], inf = consensus2(T[:, k], Cm[:, k], [c[1] for c in cands], [c[2] for c in cands], [c[3] for c in cands], n, meshes[n])
            info[n] = dict(source='consensus(base+' + '+'.join(c[0] for c in cands) + ')', **inf)
    return T, Cm, info


def best_lag(fam_pos, ok, base_pos, shifts=np.arange(-2.5, 2.501, 0.125), min_speed_cm=0.3):
    """GT-free clock check: shift s (frames) of the family position track that best matches the base on moving frames."""
    F = len(base_pos); sp = np.r_[0, np.linalg.norm(np.diff(base_pos, axis=0), axis=1) * 100]; mv = sp > min_speed_cm
    errs = []
    for s in shifts:
        i0, i1, a = idx(F, s); p = (1 - a)[:, None] * fam_pos[i0] + a[:, None] * fam_pos[i1]; m = mv & ok[i0] & ok[i1]
        errs.append(np.linalg.norm(p - base_pos, axis=1)[m].mean() * 100 if m.sum() >= 10 else np.nan)
    errs = np.array(errs)
    return (float(shifts[np.nanargmin(errs)]), float(np.nanmin(errs)), float(errs[np.argmin(np.abs(shifts))]), int(mv.sum())) if np.isfinite(errs).any() else None
