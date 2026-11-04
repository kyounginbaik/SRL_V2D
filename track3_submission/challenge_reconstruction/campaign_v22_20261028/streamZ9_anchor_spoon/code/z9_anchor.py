"""Z9: anchor frame-0 consensus rule (GT-free; one rule for every clip).

The kit aligns the whole scene with the anchor object's frame-0 pose (first
non-symmetric slot). Rule:
 1. anchor's initial static window W = frames before sustained motion
    (anchor_audit.static_window), consensus C = robust mean pose of W.
 2. candidate only if the frame-0 deviation from C is >= 2 deg or >= 10 mm
    (below that, R8 showed averaging is neutral noise).
 3. static by our own evidence: |W| >= 8, window median spread <= 2.5 deg and
    <= 5 mm, and C projected as ONE static world pose with every window frame's
    camera keeps silhouette IoU >= 0.85 x the per-frame tracked IoU on >= 80 %
    of the window (mask motion), and its stereo depth residual stays within
    10 mm of the tracked one (stereo).
 4. decision by multi-frame image evidence: the static world pose of frame 0
    (P0) and C are both evaluated on every window frame (silhouette IoU, and
    for objects with a ball model the gray-ball 2D reprojection, never ball
    depth). The window's summed cost is profiled along the geodesic
    P0 -> C (s = 0 .. 1, grid -1 .. 2); replace iff the minimiser s* is closer
    to C than to P0 (s* > 0.5) in >= 95 % of 500 frame bootstraps.
 5. replacement: the leading contiguous frames (from 0) whose deviation is
    >= the step-2 threshold get C; all other frames untouched.
Outputs: report json; with --out a new export folder (copy, never in place).
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/data1/SRL_V2D/tools/reconstruction_audit')
from anchor_audit import SYMMETRIC, mesh_path, robust_mean, static_window  # noqa: E402
from fuse import evidence  # noqa: E402

ROT_GATE, POS_GATE = 2.0, 0.010


def load_k(root, width, height):
    intr = json.loads((root / 'intrinsics.json').read_text())
    return np.array([[intr['fx'] * width / intr['width'], 0, intr['cx'] * width / intr['width']],
                     [0, intr['fy'] * height / intr['height'], intr['cy'] * height / intr['height']], [0, 0, 1.]]), intr


def frame_inputs(root, name, names, frame, width, height):
    small = lambda p: (cv2.resize(v, (width, height), interpolation=cv2.INTER_NEAREST) > 0) if (v := cv2.imread(str(p), 0)) is not None else None
    mask = small(root / 'objects' / name / 'masks/1' / f'{frame:06d}.png')
    if mask is None:
        return None
    others = [m for o in names if o != name and (m := small(root / 'objects' / o / 'masks/1' / f'{frame:06d}.png')) is not None]
    hands = [v for label in '23' if (v := small(root / 'objects' / name / 'masks' / label / f'{frame:06d}.png')) is not None]
    exclude = np.zeros((height, width), np.uint8)
    for v in others + hands:
        exclude |= v.astype(np.uint8)
    exclude = cv2.dilate(exclude, np.ones((3, 3), np.uint8)) > 0
    keep = mask.copy()
    for v in others:
        if v.sum() < mask.sum():
            keep &= ~v
    exclude &= ~keep
    enc = cv2.resize(cv2.imread(str(root / 'depth' / f'{frame:06d}.png'), -1), (width, height), interpolation=cv2.INTER_NEAREST)
    depth = 65535 / np.maximum(enc.astype(float), 1) - 1
    depth[(enc == 0) | (enc >= 65535)] = np.nan
    hand_px = int(sum(h.sum() for h in hands))
    return mask, exclude, depth, hand_px


def ball_model(root, name):
    sel = root / 'exports' / 'spoon_roll_v8' / 'selection.json'
    det = root / f'ball_detections_{name}.json'
    if name != 'wooden_spoon' or not sel.exists() or not det.exists():
        return None, None
    model = np.array(json.loads(sel.read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
    return model, json.loads(det.read_text())['frames']


def ball_px(model, balls, cam_from_obj, kfull):
    if not balls:
        return np.nan
    pts = model @ cam_from_obj[:3, :3].T + cam_from_obj[:3, 3]
    uv = pts[:, :2] / pts[:, 2:] @ kfull[:2, :2].T + kfull[:2, 2]
    return float(np.median([np.min(np.linalg.norm(uv - b['uv'], axis=1)) for b in balls]))


def analyse(root, folder, roster, mode='consensus', inject=0., width=507, height=380):
    data = dict(np.load(folder / 'object_poses.npz'))
    names = [str(n) for n in data['names']]
    order = [n for n in roster if n in names] if roster else names
    fixed = [n for n in order if n not in SYMMETRIC]
    rep = dict(root=str(root), poses=str(folder), ground_truth_used=False, anchor=fixed[0] if fixed else None, fired=False)
    if not fixed:
        rep['reason'] = 'no non-symmetric anchor (scorer uses axis rule)'
        return rep, None
    name = fixed[0]; slot = names.index(name)
    if inject > 0:  # synthetic test only: rotate the anchor's frame 0 about its centroid, seeded random axis
        rng = np.random.default_rng(int(Path(root).name[-6:]) + 7)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        m0 = trimesh.load(mesh_path(folder, name), force='mesh', process=False).centroid
        W0 = data['world_from_object'][0, slot].copy(); Rn = Rotation.from_rotvec(np.radians(inject) * ax).as_matrix()
        cw = W0[:3, :3] @ m0 + W0[:3, 3]
        W0[:3, :3] = Rn @ W0[:3, :3]; W0[:3, 3] = cw - W0[:3, :3] @ m0
        data['world_from_object'] = data['world_from_object'].copy(); data['world_from_object'][0, slot] = W0
        data['camera_from_object'] = np.linalg.inv(data['world_from_rgb'])[:, None] @ data['world_from_object']
        rep['injected_deg'] = inject; rep['injected_axis_world'] = ax.round(4).tolist()
    world = data['world_from_object'][:, slot]
    cam_w = data['world_from_rgb']
    mesh = trimesh.load(mesh_path(folder, name), force='mesh', process=False)
    centroid = np.asarray(mesh.centroid)
    R = world[:, :3, :3]; c = np.einsum('tij,j->ti', R, centroid) + world[:, :3, 3]
    end = static_window(R, c)
    rep['static_window_end'] = int(end)
    if end < 5:
        rep['reason'] = 'window < 5'
        return rep, (data['world_from_object'] if inject > 0 else None)
    Rm, cm = robust_mean(R[:end], c[:end])
    C = np.eye(4); C[:3, :3] = Rm; C[:3, 3] = cm - Rm @ centroid
    dev_r = np.degrees((Rotation.from_matrix(R[:end]) * Rotation.from_matrix(Rm).inv()).magnitude())
    dev_p = np.linalg.norm(c[:end] - cm, axis=1)
    body = (Rotation.from_matrix(Rm).inv() * Rotation.from_matrix(R[0])).as_rotvec()
    rep.update(f0_rot_dev_deg=float(dev_r[0]), f0_pos_dev_mm=float(dev_p[0] * 1000),
               f0_body_rotvec_deg=np.degrees(body).round(2).tolist(),
               window_rot_spread_deg=float(np.median(dev_r)), window_pos_spread_mm=float(np.median(dev_p) * 1000),
               per_frame_rot_dev_deg=dev_r.round(2).tolist())
    # outlier vs its own window: >= the absolute gate AND >= 3 x the window's median spread
    rgate = max(ROT_GATE, 3 * float(np.median(dev_r))) if mode == 'final' else ROT_GATE
    pgate = max(POS_GATE, 3 * float(np.median(dev_p))) if mode == 'final' else POS_GATE
    rep['gates'] = dict(rot_deg=rgate, pos_mm=pgate * 1000)
    rep['candidate'] = bool(dev_r[0] >= rgate or dev_p[0] >= pgate)
    # evidence over the window for: tracked per-frame pose, static P0, static C
    k, intr = load_k(root, width, height)
    kfull = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    points, faces = trimesh.sample.sample_surface(mesh, 20000, seed=260126)
    normals = np.asarray(mesh.face_normals)[faces]
    model, dets = ball_model(root, name)
    rows = []
    cv2.setNumThreads(2)
    for t in range(end):
        fi = frame_inputs(root, name, names, t, width, height)
        if fi is None or fi[0].sum() < 20:
            continue
        mask, exclude, depth, hand_px = fi
        inv = np.linalg.inv(cam_w[t])
        row = dict(frame=t, mask_px=int(mask.sum()), hand_px=hand_px)
        for tag, P in [('trk', world[t]), ('p0', world[0]), ('c', C)]:
            pose = inv @ P
            _, info = evidence(mesh, points, normals, pose, k, mask, exclude, depth)
            row[f'iou_{tag}'] = info['visible_iou']; row[f'z_{tag}'] = info['median_z_mm']
            if model is not None and t < len(dets):
                row[f'ball_{tag}'] = ball_px(model, dets[t], pose, kfull)
        rows.append(row)
    rep['window_rows'] = rows
    iou = {tag: np.array([r[f'iou_{tag}'] for r in rows]) for tag in ['trk', 'p0', 'c']}
    z = {tag: np.array([np.nan if r[f'z_{tag}'] is None else r[f'z_{tag}'] for r in rows]) for tag in ['trk', 'p0', 'c']}
    static_iou_ok = float(np.mean(iou['c'] >= 0.85 * iou['trk'])) if len(rows) else 0.
    zdiff = np.nanmedian(np.abs(z['c'] - z['trk'])) if np.isfinite(z['c'] - z['trk']).any() else np.nan
    rep['static'] = dict(frames=len(rows), iou_ok_fraction=static_iou_ok, depth_diff_mm=None if np.isnan(zdiff) else float(zdiff),
                         spread_ok=bool(rep['window_rot_spread_deg'] <= 2.5 and rep['window_pos_spread_mm'] <= 5.0))
    is_static = bool(end >= 8 and rep['static']['spread_ok'] and static_iou_ok >= 0.8 and (np.isnan(zdiff) or zdiff <= 10.))
    rep['static']['is_static'] = is_static
    # geodesic profile P(s) = P0 exp(s log(P0^-1 C)), s=0 frame-0 pose, s=1 consensus;
    # per-frame cost = (1 - IoU) * 10 + min(ball px, 40) / 2 (balls when a model exists)
    grid = np.linspace(-1., 2., 13)
    rel = np.linalg.inv(world[0]) @ C
    rv = Rotation.from_matrix(rel[:3, :3]).as_rotvec(); tv = rel[:3, 3]
    poses = []
    for sv in grid:
        D = np.eye(4); D[:3, :3] = Rotation.from_rotvec(sv * rv).as_matrix(); D[:3, 3] = sv * tv
        poses.append(world[0] @ D)
    prof = np.zeros((len(rows), len(grid)))
    for i, r in enumerate(rows):
        t = r['frame']
        fi = frame_inputs(root, name, names, t, width, height)
        mask, exclude, depth, _ = fi
        inv = np.linalg.inv(cam_w[t])
        for j, P in enumerate(poses):
            pose = inv @ P
            _, info = evidence(mesh, points, normals, pose, k, mask, exclude, depth)
            v = (1 - info['visible_iou']) * 10
            if model is not None and t < len(dets):
                b = ball_px(model, dets[t], pose, kfull)
                if np.isfinite(b):
                    v += min(b, 40) / 2
            prof[i, j] = v
    def argmin_s(P):
        tot = P.sum(0); j = int(np.argmin(tot))
        if 0 < j < len(grid) - 1:
            y0, y1, y2 = tot[j - 1:j + 2]; den = y0 - 2 * y1 + y2
            return float(grid[j] + (0.5 * (y0 - y2) / den if den > 0 else 0) * (grid[1] - grid[0]))
        return float(grid[j])
    s_star = argmin_s(prof)
    rng = np.random.default_rng(260128)
    boots = np.array([argmin_s(prof[rng.integers(0, len(rows), len(rows))]) for _ in range(500)])
    conf = float(np.mean(boots > 0.5)); conf0 = float(np.mean(boots > 0.))
    d = prof[:, int(np.argmin(np.abs(grid)))] - prof[:, int(np.argmin(np.abs(grid - 1.)))]
    rep['evidence'] = dict(iou_p0_mean=float(iou['p0'].mean()), iou_c_mean=float(iou['c'].mean()), iou_trk_mean=float(iou['trk'].mean()),
                           c_wins=int((d > 1e-6).sum()), p0_wins=int((d < -1e-6).sum()),
                           profile_grid=grid.round(3).tolist(), profile_cost=prof.sum(0).round(3).tolist(),
                           s_star=s_star, s_boot_p05_p95=np.percentile(boots, [5, 95]).round(3).tolist(), conf_closer_to_consensus=conf, conf_p0_rejected=conf0)
    if model is not None:
        for tag in ['p0', 'c', 'trk']:
            b = np.array([r.get(f'ball_{tag}', np.nan) for r in rows])
            rep['evidence'][f'ball_{tag}_median'] = float(np.nanmedian(b)) if np.isfinite(b).any() else None
    p = 1 - (conf0 if mode in ('evidence', 'final') else conf) + 1e-9
    rep['fired'] = bool(rep['candidate'] and is_static and p < 0.05 and mode != 'none')
    if not rep['fired']:
        rep['reason'] = 'not candidate' if not rep['candidate'] else ('window not static' if not is_static else 'evidence does not prefer consensus')
        return rep, (data['world_from_object'] if inject > 0 else None)
    lead = 0
    while lead < end and (dev_r[lead] >= rgate or dev_p[lead] >= pgate):
        lead += 1
    rep['replaced_frames'] = list(range(lead))
    new = data['world_from_object'].copy()
    if mode == 'evidence':
        sv = float(np.clip(s_star, 0., 1.))
        D = np.eye(4); D[:3, :3] = Rotation.from_rotvec(sv * rv).as_matrix(); D[:3, 3] = sv * tv
        target = world[0] @ D
    else:
        target = C
    rep['replacement_s'] = float(np.clip(s_star, 0., 1.)) if mode == 'evidence' else 1.
    new[:lead, slot] = target
    rep['applied_rot_deg'] = [float(np.degrees(Rotation.from_matrix(target[:3, :3] @ world[f, :3, :3].T).magnitude())) for f in range(lead)]
    rep['applied_pos_mm'] = [float(np.linalg.norm(target[:3, 3] - world[f, :3, 3]) * 1000) for f in range(lead)]
    return rep, new


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path); ap.add_argument('poses', type=Path)
    ap.add_argument('--roster', default=''); ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--mode', choices=['final', 'consensus', 'evidence', 'none'], default='final')
    ap.add_argument('--inject', type=float, default=0.)
    a = ap.parse_args()
    rep, new = analyse(a.root, a.poses, [r for r in a.roster.split(',') if r], a.mode, a.inject)
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        for f in a.poses.iterdir():
            if f.is_file() and f.name not in ('object_poses.npz', 'public_score.json', 'qc.json'):
                shutil.copy2(f, a.out / f.name)
            elif f.is_dir() and not (a.out / f.name).exists():
                shutil.copytree(f, a.out / f.name)
        data = dict(np.load(a.poses / 'object_poses.npz'))
        if new is not None:
            data['world_from_object'] = new
            data['camera_from_object'] = np.linalg.inv(data['world_from_rgb'])[:, None] @ new
        np.savez_compressed(a.out / 'object_poses.npz', **data)
        (a.out / 'z9_anchor.json').write_text(json.dumps({k: v for k, v in rep.items() if k != 'window_rows'}, indent=1) + '\n')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('window_rows', 'per_frame_rot_dev_deg')}), flush=True)


if __name__ == '__main__':
    main()
