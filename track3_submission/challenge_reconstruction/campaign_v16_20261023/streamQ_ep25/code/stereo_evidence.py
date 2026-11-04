"""Stream Q ep25: independent stereo-depth + silhouette evidence for board poses.
Stereo: public_gt_check_20261020/stereo/episode_000025/stereo/NNNNNN.npz (FoundationStereo depth, rectified ego_cam_c, refined
calibration), lifted exactly as project_depth.py does into the physical ego_cam_a frame, then into OUR world with the source's
world_from_rgb (the Compose RGB camera = undistorted ego_cam_a). Board pixels: per-board SAM2 masks (boards/episode_000025/{A,B}/masks/1,
A = wooden_piece_1 by ball count, Stage-A AnyCalib frame); every stereo point is projected into that frame exactly like project_depth.py
and labelled by the (3 px eroded) mask. No GT, no scans: meshes = our Compose visual meshes.
Per frame and board and per pose set: n points, median |point-to-surface| (mm), inlier fraction (< 8 mm), silhouette IoU of the projected
mesh vs the mask, and the local table height (median z of non-board stereo points within 6-20 cm of the board centre, |z| < 5 cm).
usage: stereo_evidence.py OUT.json FRAMES(e.g. 0:356:5) NAME=object_poses.npz [NAME=...]"""
import sys, json, numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
G = Path('/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'); S = G / 'stereo/episode_000025'; BD = G / 'boards/episode_000025'
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025')
ANY = G / 'outputs/episode_000025/wooden_piece_1/anycalib'
NAMES = ['wooden_piece_1', 'wooden_piece_2']; LAB = {'wooden_piece_1': 'A', 'wooden_piece_2': 'B'}
cal = json.load(open('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json'))['cameras']
K = np.array(cal['ego_cam_a']['intrinsic_matrix']); D = np.array(cal['ego_cam_a']['distortion_coefficients']); r = np.load(S / 'rectification.npz')
def mat(p): a = json.load(open(p)); return np.array([[a['fx'], 0, a['cx']], [0, a['fy'], a['cy']], [0, 0, 1.]])
Kany = mat(ANY / 'intrinsics.json'); Dany = np.array(json.load(open(ANY / 'distortion.json'))['params']); Kout = mat(ANY / 'undistorted_intrinsics.json')
T = np.linalg.inv(np.array(cal['ego_cam_a']['extrinsics_to_stereo_left'])) @ np.array(cal['ego_cam_c']['extrinsics_to_stereo_left'])
y, x = np.indices((800, 1280)); P = r['P1']; rays = np.stack([(x - P[0, 2]) / P[0, 0], (y - P[1, 2]) / P[1, 1], np.ones_like(x)], -1).reshape(-1, 3)
W_rgb = np.load(SRC / 'object_poses.npz')['world_from_rgb'].astype(float)
mesh = {n: trimesh.load(SRC / f'assets/{n}_visual.obj', force='mesh') for n in NAMES}
samp = {n: trimesh.sample.sample_surface(mesh[n], 300000, seed=0)[0] for n in NAMES}; tree = {n: cKDTree(samp[n]) for n in NAMES}
ker = np.ones((7, 7), np.uint8)
def stereo_points(k):
    z = np.load(S / f'stereo/{k:06d}.npz')['depth_m'].ravel(); v = (z > 0.15) & (z < 1.5)
    pts = rays[v] * z[v, None]; pts = pts @ r['R1']; pts = pts @ T[:3, :3].T + T[:3, 3]
    uv = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)[0]; uv = cv2.fisheye.undistortPoints(uv, Kany, Dany, P=Kout)[:, 0]
    ok = np.isfinite(uv).all(1); return pts[ok], uv[ok]
def masks(k):
    out = {}
    for n in NAMES:
        m = cv2.imread(str(BD / LAB[n] / 'masks/1' / f'{k:06d}.png'), 0); out[n] = None if m is None else (m > 0)
    return out
def silhouette(n, Tw, Wc):
    p = samp[n][::6] @ Tw[:3, :3].T + Tw[:3, 3]; c = (np.linalg.inv(Wc) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    uv = cv2.projectPoints(c, np.zeros(3), np.zeros(3), K, D)[0]; uv = cv2.fisheye.undistortPoints(uv, Kany, Dany, P=Kout)[:, 0]
    s = np.zeros((1520, 2028), np.uint8); u = np.rint(uv).astype(int); ok = (c[:, 2] > 0) & (u[:, 0] >= 0) & (u[:, 0] < 2028) & (u[:, 1] >= 0) & (u[:, 1] < 1520)
    s[u[ok, 1], u[ok, 0]] = 1; return cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
def evaluate(k, poses, pts_w, uv, M):
    row = {}
    ui = np.clip(np.rint(uv).astype(int), 0, [2027, 1519])
    anym = np.zeros(len(pts_w), bool)
    for n in NAMES:
        if M[n] is not None: anym |= M[n][ui[:, 1], ui[:, 0]]
    for n in NAMES:
        if M[n] is None: continue
        me = cv2.erode(M[n].astype(np.uint8), ker) > 0; sel = me[ui[:, 1], ui[:, 0]]; Q = pts_w[sel]
        if len(Q) > 4000: Q = Q[np.random.default_rng(k).choice(len(Q), 4000, replace=False)]
        row[n] = dict(n_points=int(sel.sum()), mask_px=int(M[n].sum()))
        for name, Tset in poses.items():
            Tw = Tset[k, NAMES.index(n)]
            if len(Q) >= 30:
                qb = (Q - Tw[:3, 3]) @ Tw[:3, :3]; d, _ = tree[n].query(qb)
                e = dict(median_mm=float(np.median(d) * 1000), inlier8=float((d < .008).mean()), p25_mm=float(np.percentile(d, 25) * 1000))
            else: e = {}
            sil = silhouette(n, Tw, W_rgb[k]); e['iou'] = float((sil & M[n]).sum() / max((sil | M[n]).sum(), 1)); e['mask_cov'] = float((sil & M[n]).sum() / max(M[n].sum(), 1))
            c = Tw[:3, :3] @ samp[n].mean(0) + Tw[:3, 3]; dxy = np.linalg.norm(pts_w[:, :2] - c[:2], axis=1)
            tb = (~anym) & (dxy > .06) & (dxy < .20) & (np.abs(pts_w[:, 2]) < .05)
            e['table_z_mm'] = float(np.median(pts_w[tb, 2]) * 1000) if tb.sum() > 50 else None
            row[n][name] = e
    return row
if __name__ == '__main__':
    out = Path(sys.argv[1]); a, b, c = map(int, sys.argv[2].split(':')); poses = {}
    for s in sys.argv[3:]:
        nm, p = s.split('=', 1); poses[nm] = np.load(p)['world_from_object']
    rows = []
    for k in range(a, min(b, 356), c):
        pts, uv = stereo_points(k); pw = pts @ W_rgb[k][:3, :3].T + W_rgb[k][:3, 3]
        rows.append(dict(frame=k, t=k / 20, **evaluate(k, poses, pw, uv, masks(k))))
        rr = rows[-1]; s = f'{k/20:5.2f}'
        for n in NAMES:
            if n not in rr: continue
            s += f' | {n[-1]} n{rr[n]["n_points"]:5d} ' + ' '.join(f'{nm}:{rr[n][nm].get("median_mm",float("nan")):4.1f}mm/{rr[n][nm].get("inlier8",float("nan")):.2f}/iou{rr[n][nm]["iou"]:.2f}' for nm in poses) + f' tbl {rr[n][list(poses)[0]]["table_z_mm"]}'
        print(s, flush=True)
    json.dump(dict(poses={k: v for k, v in (x.split('=', 1) for x in sys.argv[3:])}, rows=rows), open(out, 'w'), indent=0)
