"""G14 hedged (trust-region minimax) reference. GT-free; one fixed rule for every clip.

Representation = the kit's own: per object, pose of the registered body anchor (surface centroid of OUR mesh) + body quaternion,
and every alternative hypothesis is put into our body frame (kit _register_object_body when its mesh differs from ours) and
aligned onto the base by the kit's frame-0 scene alignment (_align_registered_scene: anchor-slot frame-0 rotation, slot-0 frame-0
position). So the hedge sees exactly the disagreement the kit would score, and frame 0 of the anchor is never moved.

Rule per object and frame (positions in cm, rotations as body-tangent rotation vectors at the base in deg; symmetric objects use the
symmetry-equivalent of each alternative nearest to the base):
  1. inliers = alternatives within the gate (|dp| <= GP cm AND |dr| <= GR deg) of the base; flips/gross (beyond 2x MP threshold)
     are never hedged (the base keeps the image / official-evidence choice).
  2. points = {base (0)} U inliers. If every point is within the margin M of the base -> no change. Else the new reference is the point
     closest to the base whose distance to every point is <= max(M, r_meb) (r_meb = minimax radius): i.e. the minimax centre when the
     hypotheses are wider than 2M apart, and the smallest move that brings all of them within M otherwise.
  3. the per-frame correction is median-filtered over SMOOTH frames (persistent disagreement only) and applied to the base.
Position and orientation are hedged separately (the MP test is separate)."""
import sys, json, os, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import median_filter
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
M = K.load_metric('MP-SR')

_mc = {}
def mesh_info(path, name):
    if path not in _mc:
        m = trimesh.load(path, force='mesh'); m.merge_vertices(); m.remove_unreferenced_vertices()
        v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        anc, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(name, v)
        raw = np.asarray(trimesh.load(path, force='mesh', process=False).vertices)
        _mc[path] = dict(v=v, f=f, anc=anc, axis=ax, order=o, raw=raw, ext=np.sort(m.extents))
    return _mc[path]

def same_mesh(pa, pb):
    a, b = np.asarray(trimesh.load(pa, force='mesh', process=False).vertices), np.asarray(trimesh.load(pb, force='mesh', process=False).vertices)
    return a.shape == b.shape and np.abs(a - b).max() < 1e-7

def T2p(T):
    q = Rot.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,))
    return np.concatenate([T[..., :3, 3], q], -1)

def to_anchor(T, anc):
    p = T2p(T); p[..., :3] += np.einsum('...ij,j->...i', T[..., :3, :3], anc); return p

def body_convert(T_alt, alt_mesh, base_mesh, name, seed):
    """alt poses (with its own mesh) -> anchor 7-vector in our body frame. Returns None if the registration is a twin / not trusted."""
    bi = mesh_info(base_mesh, name)
    if alt_mesh == base_mesh or same_mesh(alt_mesh, base_mesh):
        return to_anchor(T_alt, bi['anc']), 0.0
    v, f = budget_mesh(alt_mesh, 4096, 4096)
    ca, rot = M._register_object_body((np.asarray(v, np.float64), np.asarray(f, np.int64)), (bi['v'], bi['f']), bi['anc'], seed)
    ang = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    return M._convert_object_body(T2p(T_alt), ca, rot), ang

def align(alt7, base7, axes):
    """kit frame-0 scene alignment of the alternative onto the base. alt7/base7 [F,n,7]."""
    return M._align_registered_scene(alt7[:, None], base7[:, None], axes)[:, 0]

def nearest_sym(Rb, Ra, axis, order):
    """per frame: symmetry-equivalent of Ra nearest to Rb."""
    if not np.any(axis) or order == 1: return Ra
    rots = M._axial_rotations(axis, order)        # [o,3,3]
    cand = np.einsum('fij,ojk->foik', Ra, rots)   # Ra @ S
    rel = np.einsum('fji,fojk->foik', Rb, cand)
    ang = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    k = ang.argmin(1); return cand[np.arange(len(Ra)), k]

def meb_center(P, act, iters=400):
    """Badoiu-Clarkson minimum enclosing ball, vectorised. P [F,k,d], act [F,k] bool (row 0 = base, always active)."""
    c = P[:, 0].copy()
    for t in range(1, iters + 1):
        d = np.where(act, np.linalg.norm(P - c[:, None], axis=-1), -1)
        far = P[np.arange(len(P)), d.argmax(1)]
        c = c + (far - c) / (t + 1)
    r = np.where(act, np.linalg.norm(P - c[:, None], axis=-1), 0).max(1)
    return c, r

def closest_in_balls(P, act, rad, iters=300):
    """Dykstra projection of 0 onto the intersection of balls B(P_i, rad) (active only)."""
    F, k, d = P.shape; x = np.zeros((F, d)); inc = np.zeros((F, k, d))
    for _ in range(iters):
        for i in range(k):
            y = x + inc[:, i]
            v = y - P[:, i]; n = np.linalg.norm(v, axis=-1, keepdims=True)
            proj = np.where(n > rad[:, None], P[:, i] + v / np.maximum(n, 1e-12) * rad[:, None], y)
            proj = np.where(act[:, i:i + 1], proj, y)
            inc[:, i] = y - proj; x = proj
    return x

def hedge_points(P, act, margin):
    """P [F,k,d] with P[:,0]=0 (base). Returns correction [F,d] and per-frame minimax radius."""
    dist = np.where(act, np.linalg.norm(P, axis=-1), 0).max(1)
    c, r = meb_center(P, act)
    need = dist > margin
    out = np.zeros(P.shape[::2]) if False else np.zeros((P.shape[0], P.shape[2]))
    wide = need & (r >= margin)
    out[wide] = c[wide]
    nar = need & (r < margin)
    if nar.any():
        x = closest_in_balls(P[nar], act[nar], np.full(nar.sum(), margin * 1.0001))
        out[nar] = x
    return out, r, dist


def wgm(P, w, x, it=12):
    """vectorised weighted geometric median (Weiszfeld) per frame. P [F,k,d], w [F,k], x [F,d] start."""
    for _ in range(it):
        d = np.maximum(np.linalg.norm(P - x[:, None], axis=-1), 1e-4)
        q = w / d; x = (q[..., None] * P).sum(1) / np.maximum(q.sum(1, keepdims=True), 1e-12)
    return x

def tdist(c, P, act):
    """time-mean distance of correction track c to each hypothesis over its active frames. -> [k]"""
    d = np.linalg.norm(P - c[:, None], axis=-1)
    return (d * act).sum(0) / np.maximum(act.sum(0), 1)

def hedge_points_time(P, act, margin, minfrac=0.3, iters=200):
    """time-level trust-region minimax: smallest mean move c (per-frame track) such that every hypothesis' time-mean distance
    (over its gated frames) is <= max(margin, minimax value). Hypotheses with < minfrac gated frames are dropped."""
    F, k, d = P.shape
    act = act.astype(float).copy(); act[:, 1:] *= (act[:, 1:].mean(0) >= minfrac)
    keep = act.sum(0) > 0
    D0 = tdist(np.zeros((F, d)), P, act)
    if D0[keep].max() <= margin: return np.zeros((F, d)), dict(D0=D0.tolist(), D=D0.tolist(), rho=margin, Dstar=None)
    Nk = np.maximum(act.sum(0), 1)
    # 1) pure minimax value (mirror descent on simplex weights)
    lam = keep / keep.sum(); c = np.zeros((F, d)); best = (np.inf, c)
    for t in range(iters):
        w = lam[None] * act * F / Nk[None]
        c = wgm(P, w, c)
        D = tdist(c, P, act); m = D[keep].max()
        if m < best[0]: best = (m, c.copy())
        lam = lam * np.exp(2.0 * (D - D.max()) / max(D.max(), 1e-6)) * keep; lam /= lam.sum()
    Dstar, cmm = best
    rho = max(margin, Dstar * 1.02)
    if Dstar >= margin: return cmm, dict(D0=D0.tolist(), D=tdist(cmm, P, act).tolist(), rho=rho, Dstar=float(Dstar))
    # 2) smallest move with all time-mean distances <= rho (dual ascent; base weight 1 per frame)
    mu = np.zeros(k); c = np.zeros((F, d)); best = (np.inf, cmm)
    for t in range(iters):
        w = np.concatenate([np.ones((F, 1)), (mu[None, 1:] * act[:, 1:] * F / Nk[None, 1:])], 1)
        c = wgm(P, w, c)
        D = tdist(c, P, act)
        if D[1:][keep[1:]].max(initial=0) <= rho * 1.01:
            mv = np.linalg.norm(c, axis=-1).mean()
            if mv < best[0]: best = (mv, c.copy())
        mu[1:] = np.maximum(0, mu[1:] + 0.5 * (D[1:] - rho) / rho) * keep[1:]
    return best[1], dict(D0=D0.tolist(), D=tdist(best[1], P, act).tolist(), rho=rho, Dstar=float(Dstar))

def hedge_clip(base_T, names, base_meshes, alts, GP=6.0, GR=60.0, MP=1.0, MR=10.0, SMOOTH=9, ep=0, MODE='frame'):
    """base_T [F,n,4,4] world_from_object; alts: list of (tag, T [F,n,4,4] in base world (NaN where absent), meshes list (None=absent)).
    Returns hedged T, info."""
    F, n = base_T.shape[:2]
    infos = [mesh_info(base_meshes[s], names[s]) for s in range(n)]
    axes = [i['axis'] for i in infos]; orders = [i['order'] for i in infos]
    B7 = np.stack([to_anchor(base_T[:, s], infos[s]['anc']) for s in range(n)], 1)
    A7s, used = [], []
    for tag, T, ms in alts:
        A = B7.copy(); have = np.zeros(n, bool)
        for s in range(n):
            if ms[s] is None: continue
            Ts = T[:, s].copy(); ok = np.isfinite(Ts).all((1, 2))
            if ok.sum() < 0.5 * F: continue
            Ts[~ok] = base_T[~ok, s]
            a, ang = body_convert(Ts, ms[s], base_meshes[s], names[s], 1_000_003 * ep + 101 * s)
            if ang > 20: continue                      # registration landed on a twin: not trusted
            # frames without data -> mark NaN after alignment
            a[~ok] = np.nan; A[:, s] = a; have[s] = True
        if not have.any(): continue
        Af = np.where(np.isfinite(A), A, B7)
        Al = align(Af, B7, axes); Al[~np.isfinite(A).all(-1)] = np.nan
        for s in range(n):
            if not have[s]: Al[:, s] = np.nan
        A7s.append(Al); used.append((tag, have.tolist()))
    out = base_T.copy(); info = dict(sources=used, objects={})
    for s in range(n):
        Rb = Rot.from_quat(B7[:, s, 3:]).as_matrix()
        Pp = [np.zeros((F, 3))]; Pr = [np.zeros((F, 3))]; act = [np.ones(F, bool)]
        for Al in A7s:
            a = Al[:, s]; ok = np.isfinite(a).all(-1)
            af = np.where(ok[:, None], a, B7[:, s])
            Ra = nearest_sym(Rb, Rot.from_quat(af[:, 3:]).as_matrix(), axes[s], orders[s])
            dr = np.degrees(Rot.from_matrix(np.einsum('fji,fjk->fik', Rb, Ra)).as_rotvec())
            dp = (af[:, :3] - B7[:, s, :3]) * 100
            g = ok & (np.linalg.norm(dp, axis=-1) <= GP) & (np.linalg.norm(dr, axis=-1) <= GR)
            Pp.append(dp); Pr.append(dr); act.append(g)
        Pp, Pr, act = np.stack(Pp, 1), np.stack(Pr, 1), np.stack(act, 1)
        if MODE == 'time':
            mf = float(os.environ.get('MINFRAC', 0.3)); cp, tp = hedge_points_time(Pp, act, MP, mf); cr, tr = hedge_points_time(Pr, act, MR, mf)
            dp_ = np.where(act, np.linalg.norm(Pp, axis=-1), 0).max(1); dr_ = np.where(act, np.linalg.norm(Pr, axis=-1), 0).max(1)
        else:
            cp, rp, dp_ = hedge_points(Pp, act, MP); cr, rr, dr_ = hedge_points(Pr, act, MR); tp = tr = None
        if SMOOTH > 1:
            cp = median_filter(cp, size=(SMOOTH, 1), mode='nearest'); cr = median_filter(cr, size=(SMOOTH, 1), mode='nearest')
        # never move frame 0 (kit anchor/slot-0 alignment frame)
        cp[0] = 0; cr[0] = 0
        Rn = Rb @ Rot.from_rotvec(np.radians(cr)).as_matrix()
        pn = B7[:, s, :3] + cp / 100
        T = np.tile(np.eye(4), (F, 1, 1)); T[:, :3, :3] = Rn; T[:, :3, 3] = pn - np.einsum('fij,j->fi', Rn, infos[s]['anc'])
        out[:, s] = T
        info['objects'][names[s]] = dict(n_inlier_mean=float(act[:, 1:].sum(1).mean()), inlier_frac_per_source=[float(x) for x in act[:, 1:].mean(0)],
            move_cm_mean=float(np.linalg.norm(cp, axis=-1).mean()), move_cm_max=float(np.linalg.norm(cp, axis=-1).max()),
            move_deg_mean=float(np.linalg.norm(cr, axis=-1).mean()), move_deg_max=float(np.linalg.norm(cr, axis=-1).max()),
            spread_cm_mean=float(dp_.mean()), spread_deg_mean=float(dr_.mean()), time_pos=tp, time_rot=tr)
    return out, info
