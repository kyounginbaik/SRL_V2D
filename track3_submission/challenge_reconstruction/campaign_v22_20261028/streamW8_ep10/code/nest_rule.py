"""W8 general rule (no GT): nested-container non-penetration projection.

For every object pair (outer = larger bounding volume), a frame is a NEST frame when
  - both meshes are elongated (principal-axis variance ratio >= RATIO) so their axes are defined,
  - the inner axis is within TILT deg of the outer axis, the inner centroid is within RAD of the outer axis and
    axially inside the outer's extent, and
  - the posed meshes intersect (fcl triangle test).
Candidate pose for a NEST frame: the inner object made coaxial with the outer (rotation about its own rim centre, spin
kept), then lifted along the outer axis to the smallest non-intersecting offset (fcl bisection).
Image gate per contiguous window (>= MINLEN frames): container-opaque joint render (nestlib.render_nested) 3-class pixel
agreement against the SAM2 masks (hand masks excluded); the window is adopted only if the median per-frame change in
agreement is >= -GATE (the image evidence does not reject it). Adopted windows blend in/out over BLEND frames (SE(3)).
usage: nest_rule.py ROOT SRC_DIR OUT_DIR [--report R.json]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nestlib import *

RATIO, TILT, RAD, MINLEN, GATE, BLEND, EPS = 1.5, 20.0, 0.015, 3, 0.005, 5, 0.015


def mesh_path(pd, mj, n):
    v = mj[n]; return v if isinstance(v, str) else str(Path(pd) / v['file'])


def masks(root, names, f):
    m = {n: read_mask(root, n, f) for n in names}; hand = None
    for n in names:
        for l in '23':
            x = read_mask(root, n, f, l)
            if x is not None: hand = x if hand is None else (hand | x)
    if hand is None: hand = np.zeros_like(next(iter(m.values())))
    return m, cv2.dilate(hand.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--report'); ap.add_argument('--no-gate', action='store_true'); a = ap.parse_args()
    root, src, out = Path(a.root), Path(a.src), Path(a.out)
    P = np.load(src / 'object_poses.npz'); data = {k: P[k] for k in P.files}
    names = [str(n) for n in data['names']]; Wf = data['world_from_object'].copy(); Rg = data['world_from_rgb']
    mj = json.loads((src / 'meshes.json').read_text()); ms = {n: load_mesh(mesh_path(src, mj, n)) for n in names}
    rep = dict(root=str(root), src=str(src), ground_truth_used=False, params=dict(RATIO=RATIO, TILT=TILT, RAD=RAD, MINLEN=MINLEN, GATE=GATE, BLEND=BLEND, EPS=EPS), pairs=[])
    K, Wd, Ht = intrinsics(root); T = len(Wf)
    for io in range(len(names)):
        for ii in range(len(names)):
            if io == ii: continue
            no, ni = names[io], names[ii]; mo, mi = ms[no], ms[ni]
            if np.prod(mo.extents) < np.prod(mi.extents): continue
            ao, co, ro = container_axis(mo); ai, ci, ri = container_axis(mi)
            pr = dict(outer=no, inner=ni, ratio_outer=float(ro), ratio_inner=float(ri), nest_frames=[], windows=[])
            rep['pairs'].append(pr)
            if ro < RATIO or ri < RATIO: pr['skipped'] = 'not elongated'; continue
            ho = (np.asarray(mo.vertices) - co) @ ao; hi = (np.asarray(mi.vertices) - ci) @ ai
            rim_i = ci + ai * hi.max(); pair = Pair(mo, mi)
            nest = np.zeros(T, bool)
            for f in range(T):
                rel = np.linalg.inv(Wf[f, io]) @ Wf[f, ii]; c = rel[:3, :3] @ mi.centroid + rel[:3, 3] - co; hh = c @ ao
                rr = np.linalg.norm(c - hh * ao); tilt = np.degrees(np.arccos(np.clip((rel[:3, :3] @ ai) @ ao, -1, 1)))
                if ho.min() < hh < ho.max() and rr < RAD and tilt < TILT and pair.collide(Wf[f, io], Wf[f, ii]): nest[f] = True
            pr['nest_frames'] = [int(x) for x in np.nonzero(nest)[0]]
            if not nest.any(): continue
            cand = Wf[:, ii].copy(); lift = np.full(T, np.nan)
            for f in np.nonzero(nest)[0]:
                rel = np.linalg.inv(Wf[f, io]) @ Wf[f, ii]
                Ra = Rot.align_vectors([ao], [rel[:3, :3] @ ai])[0].as_matrix()
                rim = rel[:3, :3] @ rim_i + rel[:3, 3]; R2 = Ra @ rel[:3, :3]; t2 = rim - R2 @ rim_i
                rad = (rim - co) - ((rim - co) @ ao) * ao; t2 = t2 - rad          # rim centre onto the outer axis
                X = np.eye(4); X[:3, :3] = R2; X[:3, 3] = t2; Tc = Wf[f, io] @ X
                d = pair.min_offset(Wf[f, io], Tc, Wf[f, io][:3, :3] @ ao, dmax=0.15)
                if d is None: continue
                Tc[:3, 3] += d * (Wf[f, io][:3, :3] @ ao); cand[f] = Tc; lift[f] = d
            # windows
            idx = np.nonzero(nest & np.isfinite(lift))[0]; wins = []
            if len(idx):
                s = idx[0]; p = idx[0]
                for x in list(idx[1:]) + [None]:
                    if x is not None and x <= p + 2: p = x; continue
                    wins.append((int(s), int(p))); s = p = x
            rB, rE = Renderer([mo], K, Wd, Ht), Renderer([mi], K, Wd, Ht)
            for s, e in wins:
                w = dict(start=s, end=e, lift_cm_median=float(np.nanmedian(lift[s:e + 1]) * 100))
                if e - s + 1 < MINLEN: w['adopted'] = False; w['why'] = 'short'; pr['windows'].append(w); continue
                deltas = []
                for f in range(s, e + 1, 2):
                    if not np.isfinite(lift[f]): continue
                    M, hand = masks(root, names, f)
                    if M[no] is None or M[ni] is None: continue
                    U0 = M[no] | M[ni]
                    if U0.sum() < 200: continue
                    ys, xs = np.nonzero(U0); roi = (max(0, xs.min() - 70), max(0, ys.min() - 70), min(Wd, xs.max() + 70), min(Ht, ys.max() + 70))
                    sl = (slice(roi[1], roi[3]), slice(roi[0], roi[2])); gt = np.where(M[ni][sl], 1, np.where(M[no][sl], 0, -1)); HD = hand[sl]
                    camo = np.linalg.inv(Rg[f]) @ Wf[f, io]
                    def acc(Tw):
                        lab, _ = render_nested(rB, rE, camo, np.linalg.inv(Rg[f]) @ Tw, roi, ao, co, ho.max(), EPS)
                        U = ((lab >= 0) | (gt >= 0)) & ~HD; return float(((lab == gt) & U).sum() / max(1, U.sum()))
                    deltas.append(acc(cand[f]) - acc(Wf[f, ii]))
                w['acc_delta_median'] = float(np.median(deltas)) if deltas else None; w['n_eval'] = len(deltas)
                w['adopted'] = bool(bool(deltas) and (a.no_gate or np.median(deltas) >= -GATE))
                pr['windows'].append(w)
                if not w['adopted']: continue
                for f in range(s, e + 1):
                    if np.isfinite(lift[f]): Wf[f, ii] = cand[f]
                    else:   # hole inside window: carry the previous relative offset
                        Wf[f, ii] = Wf[f, io] @ np.linalg.inv(Wf[f - 1, io]) @ Wf[f - 1, ii]
                # blend: fade the relative correction in/out over BLEND frames outside the window
                for side, rng in (('pre', range(s - 1, max(-1, s - 1 - BLEND), -1)), ('post', range(e + 1, min(T, e + 1 + BLEND)))):
                    ref = s if side == 'pre' else e
                    D = Wf[ref, ii] @ np.linalg.inv(data['world_from_object'][ref, ii])   # world correction at the window edge
                    for k, f in enumerate(rng):
                        wgt = 1 - (k + 1) / (BLEND + 1)
                        rv = Rot.from_matrix(D[:3, :3]).as_rotvec() * wgt
                        Dk = np.eye(4); Dk[:3, :3] = Rot.from_rotvec(rv).as_matrix()
                        piv = data['world_from_object'][ref, ii][:3, 3]
                        Dk[:3, 3] = (D[:3, 3] + D[:3, :3] @ piv - piv) * wgt + piv - Dk[:3, :3] @ piv
                        Wf[f, ii] = Dk @ data['world_from_object'][f, ii]
    changed = int((np.abs(Wf - data['world_from_object']).max(axis=(2, 3)) > 1e-9).sum())
    rep['changed_object_frames'] = changed
    out.mkdir(parents=True, exist_ok=True)
    data['world_from_object'] = Wf; data['camera_from_object'] = np.linalg.inv(Rg)[:, None] @ Wf
    np.savez_compressed(out / 'object_poses.npz', **data)
    for fn in ('meshes.json', 'selection.json'):
        if (src / fn).exists():
            if fn == 'meshes.json':
                j = json.loads((src / fn).read_text()); j = {n: mesh_path(src, j, n) for n in j}   # absolute paths
                (out / fn).write_text(json.dumps(j, indent=1))
            else: shutil.copy2(src / fn, out / fn)
    (out / 'source.txt').write_text(str(src) + '\n')
    if a.report: Path(a.report).write_text(json.dumps(rep, indent=1))
    print(root.name, 'changed object-frames', changed, json.dumps([{k: v for k, v in p.items() if k != 'nest_frames'} | {'n_nest': len(p['nest_frames'])} for p in rep['pairs']]))


if __name__ == '__main__':
    main()
