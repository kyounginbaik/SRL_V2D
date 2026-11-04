"""Stream R ep32: mocap-ball evidence for the spoon roll. Detects the grey mocap balls (hue 95-150, S<35, V 120-235, compact blobs)
inside the projected reference spoon silhouette (dilated) in the undistorted RGB, lifts each blob to 3D with the stereo depth
(eval_stereo_current_20261010 depth_rgb, near percentile inside the blob), and expresses it in the spoon body frame of the given
object_poses.npz. If the reference pose is right, every ball has a constant body coordinate.
usage: spoon_markers.py OUT_JSON [--poses OBJECT_POSES_NPZ] [--overlay-frames 10,150,...]"""
import sys, json, argparse, numpy as np, cv2, trimesh
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--poses'); ap.add_argument('--overlay-frames', default=''); a = ap.parse_args()
FR = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032')
O = Path('/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000032')
DEP = Path('/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000032/depth_rgb')
z = np.load(a.poses or FR / 'object_poses.npz'); W = z['world_from_object'][:, list(z['names']).index('wooden_spoon')]; wfc = z['world_from_rgb']
K = json.load(open(O / 'blue_cup/anycalib/undistorted_intrinsics.json')); Km = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]])
sv = np.asarray(trimesh.load(FR / 'assets/wooden_spoon.obj', force='mesh').vertices)
cap = cv2.VideoCapture(str(O / 'blue_cup/anycalib/undistorted.mp4')); rows = []; ov = set(int(x) for x in a.overlay_frames.split(',') if x)
for k in range(len(W)):
    ok, im = cap.read(); assert ok
    T = np.linalg.inv(wfc[k]) @ W[k]; X = sv @ T[:3, :3].T + T[:3, 3]; q = X @ Km.T; uv = (q[:, :2] / q[:, 2:]).astype(np.int32)
    roi = np.zeros(im.shape[:2], np.uint8); cv2.fillConvexPoly(roi, cv2.convexHull(uv), 1); roi = cv2.dilate(roi, np.ones((61, 61), np.uint8))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = ((h >= 95) & (h <= 150) & (s < 35) & (v > 120) & (v < 235) & (roi > 0)).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(g)
    d = cv2.imread(str(DEP / f'{k:06d}.png'), -1).astype(np.float64); dep = 65535.0 / d - 1.0; dep[(d >= 65535) | (dep < 0.1) | (dep > 1.5)] = np.nan
    dets = []
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if not (100 < area < 2500 and 0.45 < w / hh < 2.2 and area / (w * hh) > 0.45): continue
        cxp, cyp = cen[i]; r = max(w, hh) / 2
        ys, xs = np.nonzero(lab[y:y+hh, x:x+w] == i); dd = dep[((ys + y) // 2).clip(0, 759), ((xs + x) // 2).clip(0, 1013)]; dd = dd[np.isfinite(dd)]
        if len(dd) < 5: continue
        Z = float(np.percentile(dd, 30)) + 0.006   # ball surface -> ball centre (radius ~6 mm)
        Pc = np.array([(cxp - K['cx']) * Z / K['fx'], (cyp - K['cy']) * Z / K['fy'], Z]); Pw = wfc[k][:3, :3] @ Pc + wfc[k][:3, 3]
        body = np.linalg.inv(W[k]) @ np.r_[Pw, 1]
        dets.append(dict(px=[float(cxp), float(cyp)], radius_px=float(r), depth_m=Z, body_m=body[:3].tolist(), n_depth=int(len(dd))))
        if k in ov: cv2.circle(im, (int(cxp), int(cyp)), int(r) + 6, (0, 0, 255), 3)
    rows.append(dict(frame=k, detections=dets))
    if k in ov:
        for p in uv[::7]: cv2.circle(im, tuple(p), 1, (0, 255, 0), -1)
        c = uv.mean(0).astype(int); x0, y0 = max(c[0]-300, 0), max(c[1]-300, 0); cv2.imwrite(str(Path(a.out).parent / f'markers_f{k:03d}.jpg'), im[y0:y0+600, x0:x0+600])
json.dump(dict(poses=str(a.poses or FR / 'object_poses.npz'), rows=rows), open(a.out, 'w'))
print('frames with >=1 / >=2 detections:', sum(len(r['detections']) > 0 for r in rows), sum(len(r['detections']) > 1 for r in rows))
