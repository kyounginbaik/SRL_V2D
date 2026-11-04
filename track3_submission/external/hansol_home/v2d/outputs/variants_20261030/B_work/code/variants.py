"""Variant B rules, applied identically to public/held-out (base = r8_base) and eval (base = selected20_final_v8_g10b).
GT-free: inputs are our tracker exports, our meshes and the per-type public table (built earlier from scorer outputs only).

B1 family_select: per object type, the whole track of the family that beats the base on public in BOTH mean position and mean
   orientation (paired, same objects), with zero public MP failures, n >= 3 public objects and combined gain
   dpos/3cm + dori/30deg <= -0.01; otherwise the base. Frames the family lacks / non-finite -> base.
B2 family_consensus: per frame, soft trust-region weighted SE(3) mean of the base and the type's "good" families
   (zero public MP failures for that type, paired dpos <= +0.25 cm and dori <= +3 deg). Each candidate is first aligned to
   the base: half-turn twins about our mesh's principal axes for the self-congruent types (sweeper, wooden pieces).
   Weight w = exp(-d^2/2), d = max(dp/1.5cm, dr/15deg) vs the base (continuous, so no switching jumps; candidates beyond
   ~3 sigma vanish). Symmetric types (cups, lid): position only, rotation = base. Base weight 1. Spoon and types without a
   public counterpart: base unchanged.
"""
import json, os, numpy as np
from scipy.spatial.transform import Rotation as Rot

FAMS = ['raw', 'multimap', 'pointmap_cotracker', 'bundle_adjusted']
SYMMETRIC = {'blue_cup', 'beige_cup', 'white_pot_lid'}
TWIN_TYPES = {'mini_sweeper', 'wooden_piece_1', 'wooden_piece_2'}
SIG_P, SIG_R = 1.5, 15.0


def type_rules(per_type):
    b1, good = {}, {}
    for t, fams in per_type.items():
        best = None; good[t] = []
        for f, s in fams.items():
            fam = f[4:]
            if fam not in FAMS: continue   # eval-side availability: our trackers only (frolike/fro mesh-mismatched or 4 clips)
            if s['mp_fail'] == 0 and s['dpos_vs_base'] <= 0.25 and s['dori_vs_base'] <= 3.0: good[t].append(fam)
            gain = s['dpos_vs_base'] / 3 + s['dori_vs_base'] / 30
            if s['mp_fail'] == 0 and s['n'] >= 3 and s['dpos_vs_base'] < 0 and s['dori_vs_base'] < 0 and gain <= -0.01:
                if best is None or gain < best[1]: best = (fam, gain)
        if best: b1[t] = best[0]
    return b1, good


def principal_twins(mesh_path):
    import trimesh
    m = trimesh.load(mesh_path, force='mesh', process=False)
    c = np.asarray(m.vertices).mean(0); X = np.asarray(m.vertices) - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Hs = [np.eye(4)]
    for a in Vt:
        R = Rot.from_rotvec(np.pi * a / np.linalg.norm(a)).as_matrix(); H = np.eye(4); H[:3, :3] = R; H[:3, 3] = c - R @ c; Hs.append(H)
    return Hs


def dist(A, B):
    dp = np.linalg.norm(A[..., :3, 3] - B[..., :3, 3], axis=-1) * 100
    Rr = np.einsum('...ji,...jk->...ik', A[..., :3, :3], B[..., :3, :3])
    dr = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    return dp, dr


def consensus(base, cands, name, mesh_path):
    """base (F,4,4); cands list of (F,4,4) with NaN where missing. Returns (F,4,4), info."""
    F = len(base); Hs = principal_twins(mesh_path) if name in TWIN_TYPES else [np.eye(4)]
    P = [base[:, :3, 3]]; W = [np.ones(F)]; Q = [Rot.from_matrix(base[:, :3, :3]).as_quat()]
    info = dict(candidates=len(cands), mean_weight=[], twin_frames=[])
    for C in cands:
        ok = np.isfinite(C).all((1, 2)); Cf = np.where(ok[:, None, None], C, base)
        best = None
        for h, H in enumerate(Hs):
            Ch = Cf @ H; dp, dr = dist(Ch, base); d = np.maximum(dp / SIG_P, (0 if name in SYMMETRIC else dr) / SIG_R)
            best = (Ch, d, np.full(F, h)) if best is None else (np.where((d < best[1])[:, None, None], Ch, best[0]), np.minimum(d, best[1]), np.where(d < best[1], h, best[2]))
        Ch, d, hh = best; w = np.exp(-0.5 * d ** 2) * ok
        q = Rot.from_matrix(Ch[:, :3, :3]).as_quat(); q *= np.sign((q * Q[0]).sum(1, keepdims=True) + 1e-12)
        P.append(Ch[:, :3, 3]); W.append(w); Q.append(q)
        info['mean_weight'].append(float(w.mean())); info['twin_frames'].append(int(((hh > 0) & (w > 0.01)).sum()))
    W = np.stack(W); out = base.copy()
    out[:, :3, 3] = (np.stack(P) * W[..., None]).sum(0) / W.sum(0)[:, None]
    if name not in SYMMETRIC:
        qm = (np.stack(Q) * W[..., None]).sum(0); qm /= np.linalg.norm(qm, axis=1, keepdims=True)
        out[:, :3, :3] = Rot.from_quat(qm).as_matrix()
    return out, info


def family_world(exports_dir, fam, name, world_from_rgb):
    p = os.path.join(exports_dir, fam, 'object_poses.npz')
    if not os.path.exists(p): return None
    a = np.load(p); an = [str(x) for x in a['names']]
    if name not in an: return None
    F = len(world_from_rgb); C = np.full((F, 4, 4), np.nan); Ca = a['camera_from_object'][:, an.index(name)]; Fa = min(F, len(Ca)); C[:Fa] = Ca[:Fa]
    return world_from_rgb @ C


def apply(base_npz, exports_dir, meshes, per_type_rules, variant):
    """base_npz: dict of the base object_poses.npz; meshes: name -> our mesh path. Returns new world_from_object, per-object info."""
    b1, good = per_type_rules
    names = [str(x) for x in base_npz['names']]; Wr = base_npz['world_from_rgb']; T = base_npz['world_from_object'].copy(); info = {}
    for k, n in enumerate(names):
        if variant == 'B1':
            fam = b1.get(n)
            if not fam: continue
            Fw = family_world(exports_dir, fam, n, Wr)
            if Fw is None: info[n] = dict(source='base (family missing)'); continue
            ok = np.isfinite(Fw).all((1, 2)); T[ok, k] = Fw[ok]
            info[n] = dict(source=fam, coverage=float(ok.mean()))
        else:
            fams = good.get(n, [])
            cands = [(f, family_world(exports_dir, f, n, Wr)) for f in fams]; cands = [(f, c) for f, c in cands if c is not None]
            if not cands: continue
            T[:, k], inf = consensus(T[:, k], [c for _, c in cands], n, meshes[n])
            info[n] = dict(source='consensus(base+' + '+'.join(f for f, _ in cands) + ')', **inf)
    return T, info
