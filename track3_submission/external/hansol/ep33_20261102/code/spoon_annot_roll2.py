#!/usr/bin/env python3
"""Spoon roll from a manual bowl-direction annotation, for any eval episode (generalised ep32_annot_roll.py; GT-free).

Annotation JSON (RGB clock frame -> mark): "in" = concave side away from the camera, "out" = towards the camera,
[dx, dy] = image-plane direction of the concave normal (x right, y down). Fit, per frame in [W0, W1], a roll phi about
the spoon long axis (Viterbi, 10 deg grid, phi = 0 at both ends) with
  annotation cost x4 (arrow: (err / 30 deg)^2 capped at 16; in/out: facing <= -0.6 / >= +0.6),
  filtered ball reprojection (min(px, 40) / 4), prior 0.5 (1 - cos phi), pairwise 0.04 step^2,
  optional --iou A,B,W: 30 W (1 - IoU) on frames A..B only (use where the RGB shows the spoon edge-on).
The roll is mapped to the stored (stereo) clock with the G15 delta and applied as T @ Roll(phi) to the spoon's
world_from_object and camera_from_object. Writes OUT/bundle/episode_X (copy of the base episode with the new poses),
OUT/target/episode_X (compose_target layout for score_vs_compose.py) and OUT/eval.json.
--face-thr T sets the in/out facing threshold (default 0.6 as for L3b; 0.25 keeps only the front/back sign, so the
balls decide the fine roll, e.g. a tilted spoon inside a cup marked 'in').
spoon_annot_roll2.py (2026-11-03, ep33 copy): an edge-on mark may carry a third value [dx, dy, F], the largest allowed
facing (F < 0 = more of the back side towards the camera), costed 25 max(0, facing - F) on top of the arrow term.
usage: spoon_annot_roll2.py OUT_DIR EPISODE ANN_JSON W0 W1 [--iou A,B,W] [--base BUNDLE_DIR] [--face-thr T] [--report-only]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import CleanBalls, BUNDLE, DELTA  # noqa: E402
from ep32_late_roll import roll_mat, ang, ball_px  # noqa: E402
from ep32_annot_roll import bowl_centre, bowl_dir  # noqa: E402
from guard_clip import Evidence, CAN  # noqa: E402
from spoon_hand_bowl import bowl_normal  # noqa: E402
from retime import shift  # noqa: E402

PHI = np.arange(0, 360, 10)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


THR = float(arg('--face-thr', '0.6'))  # in/out = facing <= -THR / >= +THR; 0.6 reproduces L3b, 0.25 = sign only
BALL_W = float(arg('--ball-w', '1'))  # multiplies the ball term min(px, 40) / 4


def ann_cost(a, d, fc):
    if a == 'in':
        return 25.0 * max(0.0, fc + THR), None
    if a == 'out':
        return 25.0 * max(0.0, THR - fc), None
    v = np.array(a[:2], float); v /= np.linalg.norm(v)
    e = float(np.degrees(np.arccos(np.clip(d @ v, -1, 1))))
    back = 25.0 * max(0.0, fc - a[2]) if len(a) > 2 else 0.0
    return min((e / 30.0) ** 2, 16.0) + back, e


def describe(T, b, nb, K, a):
    d, fc = bowl_dir(T, b, nb, K); _, e = ann_cost(a, d, fc)
    ok = (fc <= -THR) if a == 'in' else (fc >= THR) if a == 'out' else (e < 45 and (len(a) < 3 or fc <= a[2] + 0.05))
    return dict(facing=round(fc, 2), arrow_err=None if e is None else round(e, 0), ok=bool(ok))


def main():
    out, ep = Path(sys.argv[1]), int(sys.argv[2])
    ann = {int(k): (v if isinstance(v, str) else tuple(v)) for k, v in json.loads(Path(sys.argv[3]).read_text()).items()}
    w0, w1 = int(sys.argv[4]), int(sys.argv[5]); base = Path(arg('--base', str(BUNDLE)))
    out.mkdir(parents=True, exist_ok=True); E = f'episode_{ep:06d}'; root = CAN / 'evaluation' / E; src = base / E
    z = dict(np.load(src / 'object_poses.npz')); names = [str(n) for n in z['names']]; s = names.index('wooden_spoon')
    F = len(z['camera_from_object']); d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
    I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]])
    mj = json.loads((src / 'meshes.json').read_text()); mesh = src / mj['wooden_spoon']['file']
    m = trimesh.load(mesh, force='mesh', process=False); pts, _ = trimesh.sample.sample_surface(m, 30000, seed=5); pts = np.asarray(pts, float)
    nb = bowl_normal(str(mesh))[0]; c = m.vertices.mean(0); _, vec = np.linalg.eigh(np.cov((m.vertices - c).T)); ax = vec[:, -1]
    b = bowl_centre(pts, c, ax, nb); balls = CleanBalls(root, F)
    cam = shift(z['camera_from_object'][:, s], -d)
    print(f'ep{ep}: {len(ann)} marks; base {base.name}; annotated frame: base (facing, arrow error deg, ok)')
    for f, a in sorted(ann.items()):
        print(f, a, describe(cam[f], b, nb, K, a))
    if '--report-only' in sys.argv:
        return
    fr = list(range(w0, w1 + 1)); n = len(fr); Rm = {p: roll_mat(p, c, ax) for p in PHI}
    U = np.zeros((n, len(PHI))); ev = Evidence(root, ['wooden_spoon'], {'wooden_spoon': mesh}, F)
    if arg('--iou'):
        ia, ib, iw = arg('--iou').split(','); ia, ib, iw = int(ia), int(ib), float(iw)
        sel = [f for f in fr if ia <= f <= ib]
        for j, p in enumerate(PHI):
            iou = ev.eval('wooden_spoon', cam @ Rm[p], sel)[0]
            for f in sel:
                if np.isfinite(iou[f]):
                    U[fr.index(f), j] += 30.0 * iw * (1 - iou[f])
    for i, f in enumerate(fr):
        det = balls.det[f] if f < len(balls.det) else []
        for j, p in enumerate(PHI):
            T = cam[f] @ Rm[p]
            bp = ball_px(balls.model, T, K, det) if balls.ok else np.nan
            U[i, j] += 0 if np.isnan(bp) else BALL_W * min(bp, 40.0) / 4.0
            if f in ann:
                dd, fc = bowl_dir(T, b, nb, K); U[i, j] += ann_cost(ann[f], dd, fc)[0] * 4.0
            U[i, j] += 0.5 * (1 - np.cos(np.radians(p)))
    U[0, 1:] = U[-1, 1:] = 1e9
    D = np.zeros((n, len(PHI))); B = np.zeros((n, len(PHI)), int); D[0] = U[0]
    for i in range(1, n):
        Ra = np.stack([(cam[fr[i - 1]] @ Rm[p])[:3, :3] for p in PHI]); Rb = np.stack([(cam[fr[i]] @ Rm[p])[:3, :3] for p in PHI])
        step = np.array([[ang(x, y) for y in Rb] for x in Ra])
        tot = D[i - 1][:, None] + 0.04 * step ** 2
        B[i] = tot.argmin(0); D[i] = tot.min(0) + U[i]
    path = [int(D[-1].argmin())]
    for i in range(n - 1, 0, -1):
        path.append(B[i][path[-1]])
    phi_rgb = np.degrees(np.unwrap(np.radians(PHI[path[::-1]].astype(float))))
    full = np.zeros(F); full[w0:w1 + 1] = phi_rgb
    phi_st = np.interp(np.arange(F) + d, np.arange(F), full)
    new = dict(z)
    for key in ('camera_from_object', 'world_from_object'):
        arr = z[key].copy()
        for g in np.where(np.abs(phi_st) > 1e-9)[0]:
            arr[g, s] = arr[g, s] @ roll_mat(phi_st[g], c, ax)
        new[key] = arr
    bdir = out / 'bundle' / E
    if bdir.exists():
        shutil.rmtree(bdir)
    shutil.copytree(src, bdir)
    np.savez_compressed(bdir / 'object_poses.npz', **new)
    (bdir / 'annot_roll.json').write_text(json.dumps(dict(window_rgb=[w0, w1], iou=arg('--iou'), face_thr=THR, ball_w=BALL_W, annotation={str(k): v for k, v in ann.items()},
                                                          phi_rgb_deg=[round(float(x), 1) for x in phi_rgb],
                                                          stored_frames_changed=[int(g) for g in np.where(np.abs(phi_st) > 1e-9)[0]],
                                                          axis_body=ax.tolist(), centre_body=c.tolist(), source=str(src))))
    tdir = out / 'target' / E; tdir.mkdir(parents=True, exist_ok=True)
    for slot, nm in enumerate(names):
        np.savez_compressed(tdir / f'{nm}_trajectory.npz', T_world_obj=new['world_from_object'][:, slot])
        shutil.copy2(src / mj[nm]['file'], tdir / f'{nm}_mesh_metric.obj')
    cn = shift(new['camera_from_object'][:, s], -d)
    print('\nnew track at annotated frames')
    for f, a in sorted(ann.items()):
        print(f, a, describe(cn[f], b, nb, K, a), 'phi', round(float(full[f]), 0))
    allf = list(range(max(w0 - 3, 1), min(w1 + 4, F)))
    io = ev.eval('wooden_spoon', cam, allf)[0]; inn = ev.eval('wooden_spoon', cn, allf)[0]
    bo = np.array([ball_px(balls.model, cam[f], K, balls.det[f]) if balls.ok else np.nan for f in allf])
    bn = np.array([ball_px(balls.model, cn[f], K, balls.det[f]) if balls.ok else np.nan for f in allf])
    so = [ang(cam[f - 1][:3, :3], cam[f][:3, :3]) for f in allf]; sn = [ang(cn[f - 1][:3, :3], cn[f][:3, :3]) for f in allf]
    okc = lambda T: sum(describe(T[f], b, nb, K, a)['ok'] for f, a in ann.items())
    md = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else None
    summ = dict(episode=ep, window_rgb=[w0, w1], frames_changed_stored=int((np.abs(phi_st) > 1e-9).sum()),
                annotations_ok=dict(base=okc(cam), new=okc(cn), total=len(ann)),
                iou_mean=[float(np.nanmean(io[allf])), float(np.nanmean(inn[allf]))],
                ball_median=[md(bo), md(bn)], ball_frames=int(np.isfinite(bo).sum()),
                ball_worse_3px=int(np.nansum(bn > bo + 3)), ball_better_3px=int(np.nansum(bn < bo - 3)),
                max_step=[float(max(so)), float(max(sn))], steps_gt30=[int(sum(x > 30 for x in so)), int(sum(x > 30 for x in sn))])
    print(json.dumps(summ))
    rows = [dict(f=f, phi=float(full[f]), iou_old=float(io[f]), iou_new=float(inn[f]), ball_old=float(bo[i]), ball_new=float(bn[i]),
                 step_old=so[i], step_new=sn[i]) for i, f in enumerate(allf)]
    (out / 'eval.json').write_text(json.dumps(dict(summary=summ, rows=rows)))


if __name__ == '__main__':
    main()
