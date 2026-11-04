"""Variant A (2026-10-30): GT-free motion statistics and a robust SE(3) smoother for Track 3 object tracks.

Poses are world_from_object (F, 4, 4). Motion is measured at the mesh vertex centroid, not the object origin, so a
rotation about an off-centre origin does not count as translation. Declared symmetries mirror the public kit
(public_kit_score.SYMMETRY: cups 360, lid 2); the symmetry axis is found on OUR mesh with the kit's eigen-gap rule.
"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from scipy.spatial.transform import Rotation as Rot

SYMMETRY = {'beige_cup': 360, 'blue_cup': 360, 'white_pot_lid': 2}


def mesh_info(name, path):
    m = trimesh.load(path, force='mesh')
    v = np.asarray(m.vertices, np.float64)
    c = v.mean(0)
    axis = np.zeros(3)
    if name in SYMMETRY:
        d = v - c
        w, U = np.linalg.eigh(d.T @ d / len(d))
        gaps = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) - abs(w[(i + 1) % 3] - w[(i + 2) % 3]) for i in range(3)]
        axis = U[:, int(np.argmax(gaps))]
        axis /= np.linalg.norm(axis)
    return dict(centroid=c, axis=axis, order=SYMMETRY.get(name, 1))


def rel_angle_deg(Ra, Rb, info):
    """Symmetry-aware angle between orientations Ra[i] and Rb[i] (deg)."""
    if info['order'] == 360:
        a, b = Ra @ info['axis'], Rb @ info['axis']
        return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
    rel = np.einsum('nji,njk->nik', Ra, Rb)
    ang = np.degrees(np.linalg.norm(Rot.from_matrix(rel).as_rotvec(), axis=-1))
    if info['order'] == 2:
        flip = Rot.from_rotvec(np.pi * info['axis']).as_matrix()
        ang = np.minimum(ang, np.degrees(np.linalg.norm(Rot.from_matrix(rel @ flip).as_rotvec(), axis=-1)))
    return ang


def centers(T, info):
    return T[:, :3, :3] @ info['centroid'] + T[:, :3, 3]


def motion_stats(T, info):
    """Per-frame steps / accelerations at 20 fps units per frame."""
    p = centers(T, info) * 100.0
    R = T[:, :3, :3]
    dp = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    dr = rel_angle_deg(R[:-1], R[1:], info)
    ap = np.linalg.norm(np.diff(p, 2, axis=0), axis=-1)
    if info['order'] == 360:
        a = R @ info['axis']
        da = np.diff(a, axis=0)
        ar = np.degrees(np.linalg.norm(np.diff(da, axis=0), axis=-1))
    else:
        w = Rot.from_matrix(np.einsum('nij,nkj->nik', R[1:], R[:-1])).as_rotvec()
        ar = np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=-1))
    return dict(dp=dp, dr=dr, ap=ap, ar=ar)


def summarize(st):
    dp, dr, ap, ar = st['dp'], st['dr'], st['ap'], st['ar']
    q = lambda x, k: float(np.percentile(x, k)) if len(x) else 0.0
    return dict(rot_gt30=int((dr > 30).sum()), rot_gt15=int((dr > 15).sum()), rot_gt10=int((dr > 10).sum()),
                pos_gt3=int((dp > 3).sum()), pos_gt2=int((dp > 2).sum()), pos_gt1=int((dp > 1).sum()),
                rot_max=float(dr.max()), rot_p99=q(dr, 99), rot_p90=q(dr, 90), rot_med=q(dr, 50),
                pos_max=float(dp.max()), pos_p99=q(dp, 99), pos_p90=q(dp, 90), pos_med=q(dp, 50),
                racc_p90=q(ar, 90), racc_max=float(ar.max()) if len(ar) else 0.0, racc_gt20=int((ar > 20).sum()),
                pacc_p90=q(ap, 90), pacc_max=float(ap.max()) if len(ap) else 0.0, pacc_gt1=int((ap > 1).sum()))


# ---------------------------------------------------------------- smoother

def _whittaker(y, w, lam, order=2):
    n = len(y)
    if n <= order + 1 or lam <= 0:
        return y.copy()
    D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)) if order == 2 else sp.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n))
    A = (sp.diags(w) + lam * (D.T @ D)).tocsc()
    return spla.splu(A).solve(w[:, None] * y)


def _continuous_quats(R):
    q = Rot.from_matrix(R).as_quat()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def find_jumps(T, info, jump_deg, jump_cm, hold):
    """Persistent state changes (deliberate rolls / re-registrations) that the smoother must not average across:
    a step t->t+1 above the jump threshold after which the track does not come back near pose t within `hold` frames.
    Returns boundary indices b (segments [b_i, b_{i+1})). Steps that come back are spikes, left to the robust weights."""
    R = T[:, :3, :3]
    p = centers(T, info) * 100.0
    F = len(T)
    dr = rel_angle_deg(R[:-1], R[1:], info)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cuts = []
    for t in np.where((dr > jump_deg) | (dp > jump_cm))[0]:
        k = np.arange(t + 2, min(F, t + 2 + hold))
        if len(k) == 0:
            continue
        back_r = rel_angle_deg(np.repeat(R[t:t + 1], len(k), 0), R[k], info)
        back_p = np.linalg.norm(p[k] - p[t], axis=-1)
        came_back = np.any((back_r < 0.5 * dr[t]) & (back_p < max(0.5 * dp[t], jump_cm)))
        if not came_back:
            cuts.append(t + 1)
    return cuts


def smooth_track(T, info, lam_r, lam_p, jump_deg=60.0, jump_cm=6.0, hold=4, tukey_deg=25.0, tukey_cm=3.0,
                 iters=4, pin0=True, observed=None, unobs_w=1.0, lam_det=2.0, wmult=None):
    """Robust Whittaker (2nd-difference = acceleration penalty) smoother on the mesh-centroid trajectory and on
    sign-continuous quaternions, piecewise between persistent jumps (kept exactly as the base track has them).
    Outlier weights come from IRLS with Tukey weights at a fixed light strength lam_det (spikes get ~0 weight and are
    bridged), so a strong final lam cannot reject genuine fast motion as outliers. Frame 0 pinned (scene anchor)."""
    F = len(T)
    R = T[:, :3, :3]
    p = centers(T, info)
    cuts = find_jumps(T, info, jump_deg, jump_cm, hold)
    bounds = [0] + cuts + [F]
    Rs = R.copy()
    ps = p.copy()
    wfin = np.ones(F)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 4:
            continue
        q = _continuous_quats(R[a:b])
        y = p[a:b]
        w0 = np.ones(b - a)
        if observed is not None:
            w0 = np.where(observed[a:b], 1.0, unobs_w)
        if wmult is not None:
            w0 = w0 * wmult[a:b]
        if pin0 and a == 0:
            w0[0] = 1e6
        w = w0.copy()
        for _ in range(iters):
            qs = _whittaker(q, w, lam_det)
            qs /= np.linalg.norm(qs, axis=1, keepdims=True)
            ys = _whittaker(y, w, lam_det)
            er = rel_angle_deg(Rot.from_quat(qs).as_matrix(), R[a:b], info)
            ep = np.linalg.norm(ys - y, axis=-1) * 100.0
            u = np.maximum(er / tukey_deg, ep / tukey_cm)
            w = w0 * np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
            if wmult is not None:  # frames the image guard re-anchored are never treated as outliers
                w = np.where(wmult[a:b] > 1, w0, w)
            w = np.maximum(w, 1e-4)
            if pin0 and a == 0:
                w[0] = 1e6
        qs = _whittaker(q, w, lam_r)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        ys = _whittaker(y, w, lam_p)
        Rs[a:b] = Rot.from_quat(qs).as_matrix()
        ps[a:b] = ys
        wfin[a:b] = w
    out = T.copy()
    out[:, :3, :3] = Rs
    out[:, :3, 3] = ps - Rs @ info['centroid']
    if pin0:
        out[0] = T[0]
    return out, dict(cuts=[int(c) for c in cuts], low_weight_frames=[int(i) for i in np.where(wfin < 0.1)[0]])
