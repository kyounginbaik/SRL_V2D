"""Z9 GT-free spoon audit of a pose export (our mesh, our masks, our 2D ball detections; no ball depth).

Per frame (spoon visible, no hand near it, projection inside the image):
- 2D length check: SAM2 mask extent along the projected spoon axis vs the projected mesh extent
  (ratio, and handle-end / bowl-end offsets in mm at the end's depth);
- ball layout check: detections assigned to the 3-ball model (our pose), image distance tip<->near
  and tip<->far vs the projected model (scale/depth consistency of pose + model);
- roll-flip check: median ball px of the pose vs the pose rolled 180 deg about the body long axis
  (through the handle centre line), with >= 2 balls detected;
- silhouette IoU / depth residual come from the existing final_v5 qc.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path); ap.add_argument('poses', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=2)
    a = ap.parse_args(); root = a.root
    data = np.load(a.poses / 'object_poses.npz'); names = [str(n) for n in data['names']]
    slot = names.index('wooden_spoon'); cam = data['camera_from_object'][:, slot]; T = len(cam)
    meshes = json.loads((a.poses / 'meshes.json').read_text()); mv = meshes['wooden_spoon']
    mesh = trimesh.load(a.poses / mv['file'] if isinstance(mv, dict) else mv, force='mesh', process=False)
    pts = np.asarray(trimesh.sample.sample_surface(mesh, 30000, seed=1)[0])
    xs = pts[:, 0]; handle_end, bowl_end = xs.min(), xs.max()
    intr = json.loads((root / 'intrinsics.json').read_text()); W, H = intr['width'], intr['height']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    model = np.array(json.loads((root / 'exports/spoon_roll_v8/selection.json').read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
    dets = json.loads((root / 'ball_detections_wooden_spoon.json').read_text())['frames']
    qc = json.loads((root / 'exports/final_v5/qc.json').read_text())['objects']['wooden_spoon']['per_frame'] if (root / 'exports/final_v5/qc.json').exists() else None
    # roll flip about the body x axis through the handle centre line (y,z centre of the handle)
    hp = pts[xs < -5e-2]; yc, zc = np.median(hp[:, 1]), np.median(hp[:, 2])
    F = np.eye(4); F[:3, :3] = np.diag([1, -1, -1.]); F[:3, 3] = [0, 2 * yc, 2 * zc]
    proj = lambda P, X: (lambda c: c[:, :2] / c[:, 2:] @ K[:2, :2].T + K[:2, 2])(X @ P[:3, :3].T + P[:3, 3])
    rows = []
    for t in range(0, T, a.stride):
        P = cam[t]; row = dict(frame=t)
        if qc is not None:
            row['iou'] = qc['iou'][t]; row['z_mm'] = qc['median_z_mm'][t]
        b = dets[t] if t < len(dets) else []
        if b:
            uvd = np.array([x['uv'] for x in b]); m2 = proj(P, model); mf = proj(P @ F, model)
            dist = np.linalg.norm(m2[:, None] - uvd[None], axis=2); r, c = linear_sum_assignment(dist)
            row['ball_px'] = float(np.median(np.min(np.linalg.norm(uvd[:, None] - m2[None], axis=2), axis=1)))
            row['ball_px_flip'] = float(np.median(np.min(np.linalg.norm(uvd[:, None] - mf[None], axis=2), axis=1)))
            row['n_balls'] = len(b)
            ok = {int(i): uvd[j] for i, j in zip(r, c) if dist[i, j] < 15}
            if 2 in ok and 0 in ok:
                row['ratio_tip_near'] = float(np.linalg.norm(ok[2] - ok[0]) / np.linalg.norm(m2[2] - m2[0]))
            if 2 in ok and 1 in ok:
                row['ratio_tip_far'] = float(np.linalg.norm(ok[2] - ok[1]) / np.linalg.norm(m2[2] - m2[1]))
        mp = root / 'objects/wooden_spoon/masks/1' / f'{t:06d}.png'
        mask = cv2.imread(str(mp), 0)
        if mask is not None and mask.sum() > 0:
            mask = mask > 0
            uvm = proj(P, pts)
            inside = (uvm[:, 0] > 5).all() and (uvm[:, 0] < W - 5).all() and (uvm[:, 1] > 5).all() and (uvm[:, 1] < H - 5).all()
            near = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            hand = 0
            for lab in '23':
                hm = cv2.imread(str(root / 'objects/wooden_spoon/masks' / lab / f'{t:06d}.png'), 0)
                if hm is not None:
                    hand += int(((hm > 0) & near).sum())
            others = 0
            for o in names:
                if o != 'wooden_spoon':
                    om = cv2.imread(str(root / 'objects' / o / 'masks/1' / f'{t:06d}.png'), 0)
                    if om is not None:
                        others += int(((om > 0) & near).sum())
            row.update(mask_px=int(mask.sum()), hand_near_px=hand, other_near_px=others, inside=bool(inside))
            if inside and hand < 50 and others < 50 and mask.sum() > 800:
                a0, a1 = proj(P, np.array([[handle_end, yc, zc], [bowl_end, yc, zc]]))
                u = (a1 - a0) / max(np.linalg.norm(a1 - a0), 1e-6)
                vy, vx = np.nonzero(mask); s_mask = np.stack([vx, vy], 1) @ u
                s_mesh = uvm @ u
                lo_m, hi_m = np.percentile(s_mask, [0.5, 99.5]); lo_r, hi_r = s_mesh.min(), s_mesh.max()
                zh = (P[:3, :3] @ np.array([handle_end, yc, zc]) + P[:3, 3])[2]; zb = (P[:3, :3] @ np.array([bowl_end, yc, zc]) + P[:3, 3])[2]
                foreshort = np.linalg.norm(a1 - a0) / (K[0, 0] * (bowl_end - handle_end) / ((zh + zb) / 2))
                if foreshort > 0.6:
                    row['len_ratio'] = float((hi_m - lo_m) / (hi_r - lo_r))
                    row['handle_end_mm'] = float((lo_r - lo_m) * zh / K[0, 0] / foreshort * 1000)  # + = mask longer than mesh
                    row['bowl_end_mm'] = float((hi_m - hi_r) * zb / K[0, 0] / foreshort * 1000)
                    row['foreshortening'] = float(foreshort)
        rows.append(row)
    def stat(key, f=np.median):
        v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])])
        return (float(f(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), int(len(v))) if len(v) else None
    flips = [r['frame'] for r in rows if r.get('n_balls', 0) >= 2 and r['ball_px'] > 15 and r['ball_px_flip'] < 0.5 * r['ball_px']]
    lowiou = [r['frame'] for r in rows if r.get('iou') is not None and r['iou'] < 0.3]
    summary = dict(mesh_length_cm=float((bowl_end - handle_end) * 100), mesh_bowl_width_cm=float(np.ptp(pts[xs > .07][:, 1]) * 100),
                   mesh_bowl_thickness_cm=float(np.ptp(pts[xs > .07][:, 2]) * 100), handle_axis_yz_cm=[float(yc * 100), float(zc * 100)],
                   tip_ball_inside_handle_end_mm=float((model[2, 0] - handle_end) * 1000),
                   len_ratio=stat('len_ratio'), handle_end_mm=stat('handle_end_mm'), bowl_end_mm=stat('bowl_end_mm'),
                   ball_ratio_tip_near=stat('ratio_tip_near'), ball_ratio_tip_far=stat('ratio_tip_far'),
                   ball_px=stat('ball_px'), ball_px_flip=stat('ball_px_flip'), z_mm=stat('z_mm'), iou=stat('iou'),
                   likely_roll_flip_frames=flips, low_iou_frames=lowiou, frames_sampled=len(rows), stride=a.stride)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(root=str(root), poses=str(a.poses), ground_truth_used=False, ball_depth_used=False, summary=summary, rows=rows)) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
