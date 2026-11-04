"""Shared paths / loaders for the ComPose-on-V2D pipeline (runs inside the v2d container: /jisu, /shared mounts)."""
import json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/jisu/NVIDIA_CHALLENGE_claude')
COMPOSE = Path('/shared/NeurIPS_Tracking/ComPose_final')
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'compose_v2d'))

from v2d_io import PUBLIC_EPISODES, T3, cam_params, load_calib, video_path  # noqa: E402


def split_of(ep):
    return 'public' if ep in PUBLIC_EPISODES else 'evaluation'


def work_dir(ep, obj):
    d = ROOT / 'outputs' / f'compose_ep{ep}_{obj}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def mono_masks_dir(ep, obj):
    """SAM3 video masks tracked on the rect-left mono stream (sam3_track_mono.py convention)."""
    d = ROOT / 'outputs' / f'mono_masks_ep{ep}'
    if obj != 'white_pot' or not d.exists():
        d2 = ROOT / 'outputs' / f'mono_masks_ep{ep}_{obj}'
        if d2.exists():
            d = d2
    return d


def load_depth(ep):
    z = np.load(ROOT / 'depth' / split_of(ep) / f'episode_{ep:06d}' / 'depth_mm.npz')
    meta = json.load(open(ROOT / 'depth' / split_of(ep) / f'episode_{ep:06d}' / 'meta.json'))
    return z['depth_mm'], z['frames'].tolist(), meta


def load_T_a_from_rectL(ep, depth_meta):
    """4x4: rect-left points -> ego_cam_a camera frame. Prefer the per-episode PnP-refined one."""
    mp = ROOT / 'data/first_frames' / split_of(ep) / f'episode_{ep:06d}' / 'pointmap_cam_a_f0_meta.json'
    if mp.exists():
        m = json.load(open(mp))
        if 'T_a_from_rectL_frame' in m:
            return np.array(m['T_a_from_rectL_frame'], np.float64), 'frame_pnp'
    return np.array(depth_meta['T_a_from_rectL_factory_refinedstereo'], np.float64), 'factory'


def backproject(depth_mm, mask, K):
    """depth (H,W) uint16 mm, mask (H,W) bool -> (M,3) metres + (v,u) pixel indices."""
    v, u = np.nonzero(mask & (depth_mm > 0))
    Z = depth_mm[v, u].astype(np.float64) / 1000.0
    P = np.stack([(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z], 1)
    return P, v, u


def load_meta(ep, obj):
    return json.load(open(work_dir(ep, obj) / 'meta.json'))


def load_ego(ep, obj):
    """(N,4,4) cam0_T_cam(f) in the rect-left frame, from stereo_vo.py."""
    z = np.load(work_dir(ep, obj) / 'ego_cam0_T_cam.npz')
    return z['cam0_T_cam'], z['ok']


def gt_world_T_obj(ep, obj):
    """GT mocap trajectory of `obj` (public episodes): (N,4,4) world_T_obj + visible flags."""
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation as Rot
    t = pq.read_table(T3 / f'public/data/chunk-000/episode_{ep:06d}.parquet').to_pandas()
    rows = [[o for o in r if o['name'] == obj][0] for r in t['observation.objects']]
    poses = np.array([r['pose'] for r in rows], np.float64)        # x y z qw qx qy qz
    vis = np.array([bool(r['visible']) for r in rows])
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :3] = Rot.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    T[:, :3, 3] = poses[:, :3]
    return T, vis


def rot_err_deg(R1, R2):
    from scipy.spatial.transform import Rotation as Rot
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(np.asarray(R1) @ np.asarray(R2).T).as_rotvec())))


def depth_connected_core(mask, depth_mm, jump_mm=15, min_frac=0.02):
    """Copy of scripts/build_first_frames.depth_connected_core (avoids importing SAM3 there):
    keep depth-continuous components (4-neighbours within jump_mm) that are large enough."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    H, W = mask.shape
    valid = mask & (depth_mm > 0)
    n = int(valid.sum())
    if n == 0:
        return np.zeros((H, W), bool)
    idx = -np.ones((H, W), np.int64); idx[valid] = np.arange(n)
    d = depth_mm.astype(np.int32)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = valid[: H - dy, : W - dx] & valid[dy:, dx:] & (np.abs(d[: H - dy, : W - dx] - d[dy:, dx:]) < jump_mm)
        rows.append(idx[: H - dy, : W - dx][a]); cols.append(idx[dy:, dx:][a])
    g = coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    keep = np.isin(lab, np.nonzero(sizes >= max(50, min_frac * n))[0])
    out = np.zeros((H, W), bool); out[valid] = keep
    return out


def transfer_mask(mask_rgb, depth_mm, K_rect, T_a_rect, Ka, Da, cell=4, z_tol=0.03):
    """Copy of scripts/build_first_frames.transfer_mask: raw-RGB mask -> rect-left mask via the stereo depth, with a
    z-buffer in the RGB view so background seen past the silhouette from the mono viewpoint is rejected."""
    H, W = depth_mm.shape
    v, u = np.nonzero(depth_mm > 0)
    Z = depth_mm[v, u] / 1000.0
    f, cx, cy = K_rect[0, 0], K_rect[0, 2], K_rect[1, 2]
    P = np.stack([(u - cx) * Z / f, (v - cy) * Z / f, Z], 1)
    Pa = P @ T_a_rect[:3, :3].T + T_a_rect[:3, 3]
    ok = Pa[:, 2] > 0.05
    xn, yn = Pa[:, 0] / np.maximum(Pa[:, 2], 1e-6), Pa[:, 1] / np.maximum(Pa[:, 2], 1e-6)
    ok &= (np.abs(xn) < 1.6) & (np.abs(yn) < 1.2)
    out = np.zeros((H, W), bool)
    if not ok.any():
        return out
    p = cv2.projectPoints(P[ok], cv2.Rodrigues(T_a_rect[:3, :3])[0], T_a_rect[:3, 3], Ka, Da)[0].reshape(-1, 2)
    x, y = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
    Hm, Wm = mask_rgb.shape
    inside = (x >= 0) & (x < Wm) & (y >= 0) & (y < Hm)
    za = Pa[ok, 2]
    gx, gy = x[inside] // cell, y[inside] // cell
    zbuf = np.full((Hm // cell + 1, Wm // cell + 1), np.inf)
    np.minimum.at(zbuf, (gy, gx), za[inside])
    front = np.zeros(ok.sum(), bool); front[inside] = za[inside] <= zbuf[gy, gx] + z_tol
    hit = np.zeros(ok.sum(), bool); hit[inside] = mask_rgb[y[inside], x[inside]]
    idx = np.nonzero(ok)[0][hit & front]
    out[v[idx], u[idx]] = True
    return out


def gt_mesh_path(obj):
    """Public GT scan of `obj` (file names vary per object); None when the object has no public scan (e.g. wooden_spoon)."""
    import glob as _glob
    d = T3 / 'public' / 'mesh' / obj
    for pat in (f'{obj}.glb', f'{obj}_visual.glb', '*visual*', '*.glb', '*.obj', '*.ply', '*.stl'):
        c = [x for x in sorted(_glob.glob(str(d / pat))) if 'collision' not in x]
        if c:
            return c[0]
    return None
