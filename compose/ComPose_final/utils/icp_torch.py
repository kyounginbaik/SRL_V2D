"""
GPU-batched ICP + centroid using PyTorch3D.
Fully batched, no for loops.
Supports color-aware ICP by appending weighted RGB to XYZ.
"""

import torch
import numpy as np
from pytorch3d.ops import iterative_closest_point


def batch_subsample(pts, mask, n_samples):
    """Batched random subsampling. Invalid points pushed to end."""
    B, K, D = pts.shape
    rand_idx = torch.rand(B, K, device=pts.device)
    rand_idx[~mask] = 2.0  # invalid → end
    _, sorted_idx = rand_idx.sort(dim=1)
    idx = sorted_idx[:, :n_samples]
    pts_sub = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    mask_sub = torch.gather(mask, 1, idx)
    return pts_sub, mask_sub


def batch_outlier_removal(pts, mask, sigma=2.0):
    """Batched outlier removal. Only uses first 3 dims (xyz) for distance."""
    xyz = pts[..., :3]
    xyz_z = xyz.clone()
    xyz_z[~mask] = 0
    cnt = mask.float().sum(1, keepdim=True).clamp(min=1)
    center = xyz_z.sum(1, keepdim=True) / cnt.unsqueeze(-1)
    dist = (xyz - center).norm(dim=-1)
    dist[~mask] = 0
    d_mean = (dist * mask.float()).sum(1, keepdim=True) / cnt
    d_std = (((dist**2 * mask.float()).sum(1, keepdim=True) / cnt - d_mean**2).clamp(min=0)).sqrt()
    mask = mask & (dist < d_mean + sigma * d_std)
    return mask


def batch_icp(pts_i, pts_j, mask_i, mask_j, max_points=500, max_iter=50,
              color_weight=0.0, min_points=10):
    """Fully batched ICP using PyTorch3D.

    Args:
        pts_i/j: (B, K, 3) or (B, K, 6) if color appended [xyz + rgb in 0-1]
        mask_i/j: (B, K) bool
        color_weight: weight for color channels in distance. 0 = geometry only.
        min_points: if either mask has fewer valid points, return identity R and zero T
                    for that batch element (prevents NaN from degenerate ICP).
    Returns:
        R: (B, 3, 3)
        T: (B, 3) translation in original (uncentered) space
    """
    B = pts_i.shape[0]
    device = pts_i.device
    has_color = pts_i.shape[-1] == 6 and color_weight > 0

    # Outlier removal (uses xyz only)
    mask_i = batch_outlier_removal(pts_i, mask_i)
    mask_j = batch_outlier_removal(pts_j, mask_j)

    # Track validity per batch element: both sides must have enough points
    valid = (mask_i.sum(1) >= min_points) & (mask_j.sum(1) >= min_points)  # (B,)

    # Subsample
    pi, mi = batch_subsample(pts_i, mask_i, max_points)
    pj, mj = batch_subsample(pts_j, mask_j, max_points)

    # Split xyz and color
    pi_xyz = pi[..., :3]
    pj_xyz = pj[..., :3]

    # Center each sample (xyz only)
    pi_z = pi_xyz.clone(); pi_z[~mi] = 0
    pj_z = pj_xyz.clone(); pj_z[~mj] = 0
    cnt_i = mi.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    cnt_j = mj.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    ci = pi_z.sum(1, keepdim=True) / cnt_i  # (B, 1, 3)
    cj = pj_z.sum(1, keepdim=True) / cnt_j

    pi_c = pi_xyz - ci
    pj_c = pj_xyz - cj
    pi_c[~mi] = 0
    pj_c[~mj] = 0

    # Build ICP input: optionally append weighted color
    if has_color:
        pi_rgb = pi[..., 3:6] * color_weight
        pj_rgb = pj[..., 3:6] * color_weight
        pi_rgb[~mi] = 0
        pj_rgb[~mj] = 0
        icp_i = torch.cat([pi_c, pi_rgb], dim=-1)  # (B, K, 6)
        icp_j = torch.cat([pj_c, pj_rgb], dim=-1)
    else:
        icp_i = pi_c
        icp_j = pj_c

    # PyTorch3D ICP
    result = iterative_closest_point(
        icp_i, icp_j,
        estimate_scale=False,
        max_iterations=max_iter,
    )
    # pytorch3d: X @ R + T = Y  →  we want R @ X convention
    R_full = result.RTs.R.transpose(-1, -2)  # (B, D, D) where D=3 or 6
    T_icp = result.RTs.T  # (B, D) in centered space

    # Extract 3x3 rotation and 3D translation only
    R = R_full[:, :3, :3]
    T_icp_xyz = T_icp[:, :3]

    # Convert T back to original space: T_orig = cj - R @ ci + T_icp
    ci_sq = ci.squeeze(1)  # (B, 3)
    cj_sq = cj.squeeze(1)
    T = cj_sq - (R @ ci_sq.unsqueeze(-1)).squeeze(-1) + T_icp_xyz

    # Replace degenerate/NaN entries with identity R and zero T
    bad = ~valid | torch.isnan(R).any(dim=(1, 2)) | torch.isnan(T).any(dim=1)
    if bad.any():
        eye = torch.eye(3, device=device, dtype=R.dtype).expand(B, -1, -1)
        R = torch.where(bad[:, None, None], eye, R)
        T = torch.where(bad[:, None], torch.zeros_like(T), T)

    return R, T


def batch_icp_rotation(pts_i, pts_j, mask_i, mask_j, max_points=500, max_iter=50):
    """Backward-compatible wrapper that returns R only."""
    R, _ = batch_icp(pts_i, pts_j, mask_i, mask_j,
                     max_points=max_points, max_iter=max_iter)
    return R


def batch_centroid(pts, mask):
    """Batched centroid + std. No for loops. Uses xyz only (first 3 dims).
    Empty-mask samples (all False) return zeros rather than NaN.
    """
    xyz = pts[..., :3]
    mask = batch_outlier_removal(pts, mask)
    pts_z = xyz.clone()
    pts_z[~mask] = 0
    cnt = mask.float().sum(1, keepdim=True).clamp(min=1)
    centroids = pts_z.sum(1) / cnt  # (B, 3)

    diff = xyz - centroids.unsqueeze(1)
    diff[~mask] = 0
    var = (diff**2 * mask.float().unsqueeze(-1)).sum(1) / cnt
    stds = var.sum(-1).sqrt()

    # Empty-mask samples: zero out (avoid NaN propagation)
    empty = mask.sum(1) == 0  # (B,)
    if empty.any():
        centroids = torch.where(empty[:, None], torch.zeros_like(centroids), centroids)
        stds = torch.where(empty, torch.zeros_like(stds), stds)

    return centroids, stds


def pad_pointclouds(pc_list, max_k=5000):
    """Pad variable-length pointclouds to fixed size.
    Supports (K, 3) xyz or (K, 6) xyzrgb.
    """
    N = len(pc_list)
    # Detect dimensionality from first non-empty entry
    dim = 3
    for pc in pc_list:
        if isinstance(pc, np.ndarray) and len(pc) > 0:
            dim = pc.shape[-1]; break
        elif isinstance(pc, torch.Tensor) and len(pc) > 0:
            dim = pc.shape[-1]; break

    pts = torch.zeros(N, max_k, dim)
    mask = torch.zeros(N, max_k, dtype=torch.bool)
    for i, pc in enumerate(pc_list):
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc)
        k = min(len(pc), max_k)
        if k > 0:
            pts[i, :k] = pc[:k]
            mask[i, :k] = True
    return pts, mask
