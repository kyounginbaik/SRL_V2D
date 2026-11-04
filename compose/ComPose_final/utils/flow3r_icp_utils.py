"""
Shared utilities for Flow3R-based object ICP across all datasets.
Runs Flow3R on entire sequence at once → extracts per-frame 3D pointmaps → ICP between consecutive pairs.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as tvf

_transform = tvf.ToTensor()


def resize_14x(img, max_long=518):
    """Resize keeping aspect ratio, dimensions to multiples of 14."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = max_long / max(h, w)
    new_w = max(14, int(round(w * scale / 14)) * 14)
    new_h = max(14, int(round(h * scale / 14)) * 14)
    return cv2.resize(img, (new_w, new_h))


def _frame_centroid(pts):
    """Compute absolute centroid + std from one frame's masked pointcloud."""
    pts = _clean_points(pts)
    if len(pts) < 10:
        return np.zeros(3, dtype=np.float32), np.float32(0.0)
    centroid = pts.mean(0).astype(np.float32)
    std = np.float32(np.linalg.norm(pts.std(axis=0)))
    return centroid, std


def flow3r_extract_pointclouds(images_rgb, masks, flow3r_model, device,
                               max_long=518, chunk_size=80, max_points=5000):
    """
    Run Flow3R on sequence → extract per-frame masked pointclouds.

    Args:
        max_points: subsample to this many points per frame (0=no limit)

    Returns:
        pointclouds: list of (K_i, 3) float32 np arrays — cleaned pointcloud per frame
    """
    N = len(images_rgb)
    if N < 1:
        return []

    imgs_resized = [resize_14x(img, max_long=max_long) for img in images_rgb]
    rH, rW = imgs_resized[0].shape[:2]
    masks_resized = [cv2.resize(m.astype(np.uint8), (rW, rH),
                                interpolation=cv2.INTER_NEAREST) for m in masks]

    imgs_tensor = torch.stack([_transform(img) for img in imgs_resized])

    all_pointmaps = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = imgs_tensor[start:end].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = flow3r_model(chunk)
        all_pointmaps.append(pred['local_points'][0].float().cpu().numpy())

    pointmaps = np.concatenate(all_pointmaps, axis=0)  # (N, rH, rW, 3)

    pointclouds = []
    for i in range(N):
        pts = pointmaps[i][masks_resized[i] > 0]
        pts = _clean_points(pts)
        if max_points > 0 and len(pts) > max_points:
            pts = pts[np.random.choice(len(pts), max_points, replace=False)]
        pointclouds.append(pts.astype(np.float32))

    return pointclouds


def vggt_sequence_icp(images_rgb, masks, vggt_model, device,
                      max_long=518, chunk_size=16):
    """
    Run VGGT on sequence, extract world-space pointmaps, compute per-pair ICP.

    Args:
        images_rgb: list of (H, W, 3) RGB numpy arrays
        masks: list of (H, W) uint8/bool object masks
        vggt_model: loaded VGGT model
        device: torch device
        max_long: max dimension for resize
        chunk_size: max frames per VGGT forward

    Returns:
        R_objs: dict {frame_idx: (3,3) rotation matrix}
        T_objs: dict {frame_idx: (3,) translation delta}
    """
    N = len(images_rgb)
    if N < 2:
        return {}, {}

    # Resize all frames
    imgs_resized = [resize_14x(img, max_long=max_long) for img in images_rgb]
    rH, rW = imgs_resized[0].shape[:2]
    masks_resized = [cv2.resize(m.astype(np.uint8), (rW, rH),
                                interpolation=cv2.INTER_NEAREST) for m in masks]

    # Stack to tensor (VGGT expects (B, S, 3, H, W) in [0, 1])
    imgs_tensor = torch.stack([_transform(img) for img in imgs_resized])  # (N, 3, rH, rW)

    # Run VGGT in chunks
    all_pointmaps = []
    all_confs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = imgs_tensor[start:end].unsqueeze(0).to(device)  # (1, C, 3, rH, rW)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            preds = vggt_model(images=chunk)
        pm = preds['world_points'][0].float().cpu().numpy()  # (C, rH, rW, 3)
        conf = preds['world_points_conf'][0].float().cpu().numpy()  # (C, rH, rW, 1)
        all_pointmaps.append(pm)
        all_confs.append(conf)

    pointmaps = np.concatenate(all_pointmaps, axis=0)  # (N, rH, rW, 3)
    confmaps = np.concatenate(all_confs, axis=0)[..., 0]  # (N, rH, rW)

    # ICP for consecutive pairs
    from concurrent.futures import ThreadPoolExecutor
    conf_thr = np.median(confmaps[confmaps > 0]) * 0.5  # adaptive threshold

    def _icp_pair(i):
        mask_i = (masks_resized[i] > 0) & (confmaps[i] > conf_thr)
        mask_j = (masks_resized[i + 1] > 0) & (confmaps[i + 1] > conf_thr)
        pts_i = pointmaps[i][mask_i]
        pts_j = pointmaps[i + 1][mask_j]
        R = icp_rotation(pts_i, pts_j)
        # Translation: full points (mask only, no conf filter) for stable centroid
        pts_i_full = pointmaps[i][masks_resized[i] > 0]
        pts_j_full = pointmaps[i + 1][masks_resized[i + 1] > 0]
        t = _centroid_delta(pts_i_full, pts_j_full)
        return i, R, t

    R_objs = {}
    T_objs = {}
    with ThreadPoolExecutor(max_workers=min(16, N - 1)) as pool:
        for i, R, t in pool.map(lambda i: _icp_pair(i), range(N - 1)):
            R_objs[i] = R
            T_objs[i] = t

    return R_objs, T_objs


def _clean_points(pts):
    """Filter invalid + outlier removal. Returns cleaned points."""
    valid = np.isfinite(pts).all(axis=-1) & (pts[:, 2] > 0)
    pts = pts[valid]
    if len(pts) < 10:
        return pts
    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=-1)
    thr = np.median(dist) + 2.0 * dist.std()
    keep = dist < thr
    if keep.sum() >= 10:
        pts = pts[keep]
    return pts


def _centroid_delta(pi, pj):
    """Compute translation delta + per-frame std from full pointclouds."""
    _default = (np.zeros(3, dtype=np.float32), np.float32(0.0), np.float32(0.0))
    if len(pi) < 10 or len(pj) < 10:
        return _default
    pi = _clean_points(pi)
    pj = _clean_points(pj)
    if len(pi) < 10 or len(pj) < 10:
        return _default
    delta = (pj.mean(0) - pi.mean(0)).astype(np.float32)
    std_i = np.float32(np.linalg.norm(pi.std(axis=0)))
    std_j = np.float32(np.linalg.norm(pj.std(axis=0)))
    return delta, std_i, std_j


def icp_rotation(pi, pj, max_points=500, return_translation=False, use_median_centroid=False):
    """Compute rotation (and optionally translation) between two point clouds via ICP."""
    import open3d as o3d
    _default = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)) if return_translation else np.eye(3, dtype=np.float32)
    if len(pi) < 10 or len(pj) < 10:
        return _default

    # Subsample FIRST (before any expensive ops)
    if len(pi) > max_points * 2:
        pi = pi[np.random.choice(len(pi), max_points * 2, replace=False)]
    if len(pj) > max_points * 2:
        pj = pj[np.random.choice(len(pj), max_points * 2, replace=False)]

    # Filter invalid
    valid_i = np.isfinite(pi).all(axis=-1) & (pi[:, 2] > 0)
    valid_j = np.isfinite(pj).all(axis=-1) & (pj[:, 2] > 0)
    pi, pj = pi[valid_i], pj[valid_j]
    if len(pi) < 10 or len(pj) < 10:
        return _default

    # Outlier removal on subsampled set
    for arr_name in ['pi', 'pj']:
        pts = pi if arr_name == 'pi' else pj
        med = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - med, axis=-1)
        thr = np.median(dist) + 2.0 * dist.std()
        keep = dist < thr
        if keep.sum() >= 10:
            if arr_name == 'pi':
                pi = pts[keep]
            else:
                pj = pts[keep]

    if len(pi) < 10 or len(pj) < 10:
        return _default

    # Final subsample to max_points
    if len(pi) > max_points:
        pi = pi[np.random.choice(len(pi), max_points, replace=False)]
    if len(pj) > max_points:
        pj = pj[np.random.choice(len(pj), max_points, replace=False)]

    if use_median_centroid:
        ci, cj = np.median(pi, axis=0), np.median(pj, axis=0)
    else:
        ci, cj = pi.mean(0), pj.mean(0)
    pci = o3d.geometry.PointCloud()
    pci.points = o3d.utility.Vector3dVector((pi - ci).astype(np.float64))
    pcj = o3d.geometry.PointCloud()
    pcj.points = o3d.utility.Vector3dVector((pj - cj).astype(np.float64))
    reg = o3d.pipelines.registration.registration_icp(
        pci, pcj, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    R = np.array(reg.transformation)[:3, :3].astype(np.float32)
    if return_translation:
        # Translation delta = centroid_j - centroid_i
        t_delta = (cj - ci).astype(np.float32)
        return R, t_delta
    return R
