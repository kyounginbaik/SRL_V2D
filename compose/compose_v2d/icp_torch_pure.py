"""Pure-torch replacement for ComPose's utils/icp_torch.py (which needs pytorch3d.ops.iterative_closest_point).

Same signatures as the original: batch_icp / batch_icp_rotation / batch_centroid / pad_pointclouds.
Nearest neighbours via torch.cdist on the subsampled (<=500 pt) clouds, Kabsch update, fixed iteration count.
Returns R in the "R @ X" convention (same as the original after its transpose) and T in the original space.
"""
import numpy as np
import torch


def batch_subsample(pts, mask, n_samples):
    B, K, D = pts.shape
    rand_idx = torch.rand(B, K, device=pts.device)
    rand_idx[~mask] = 2.0
    _, sorted_idx = rand_idx.sort(dim=1)
    idx = sorted_idx[:, :n_samples]
    pts_sub = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    mask_sub = torch.gather(mask, 1, idx)
    return pts_sub, mask_sub


def batch_outlier_removal(pts, mask, sigma=2.0):
    xyz = pts[..., :3]
    xyz_z = xyz.clone(); xyz_z[~mask] = 0
    cnt = mask.float().sum(1, keepdim=True).clamp(min=1)
    center = xyz_z.sum(1, keepdim=True) / cnt.unsqueeze(-1)
    dist = (xyz - center).norm(dim=-1); dist[~mask] = 0
    d_mean = (dist * mask.float()).sum(1, keepdim=True) / cnt
    d_std = (((dist ** 2 * mask.float()).sum(1, keepdim=True) / cnt - d_mean ** 2).clamp(min=0)).sqrt()
    return mask & (dist < d_mean + sigma * d_std)


def _kabsch(P, Q, w):
    """Weighted Kabsch: R minimising sum w |R p - q|^2 for centred P, Q (B, K, 3), w (B, K)."""
    w_ = w.unsqueeze(-1)
    H = (w_ * P).transpose(1, 2) @ Q  # (B, 3, 3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2))
    D = torch.ones_like(S); D[:, -1] = torch.where(d < 0, -1.0, 1.0)
    R = Vt.transpose(-1, -2) @ torch.diag_embed(D) @ U.transpose(-1, -2)
    return R


def batch_icp(pts_i, pts_j, mask_i, mask_j, max_points=500, max_iter=50,
              color_weight=0.0, min_points=10):
    """Fully batched point-to-point ICP. pts: (B, K, 3|6), mask: (B, K) bool.
    Returns R (B,3,3) with  pts_j ~ R @ pts_i + T,  T (B,3) in original (uncentred) space."""
    B = pts_i.shape[0]; device = pts_i.device
    has_color = pts_i.shape[-1] == 6 and color_weight > 0
    mask_i = batch_outlier_removal(pts_i, mask_i)
    mask_j = batch_outlier_removal(pts_j, mask_j)
    valid = (mask_i.sum(1) >= min_points) & (mask_j.sum(1) >= min_points)

    pi, mi = batch_subsample(pts_i, mask_i, max_points)
    pj, mj = batch_subsample(pts_j, mask_j, max_points)
    pi_xyz, pj_xyz = pi[..., :3], pj[..., :3]
    cnt_i = mi.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    cnt_j = mj.float().sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    pi_z = pi_xyz.clone(); pi_z[~mi] = 0
    pj_z = pj_xyz.clone(); pj_z[~mj] = 0
    ci = pi_z.sum(1, keepdim=True) / cnt_i
    cj = pj_z.sum(1, keepdim=True) / cnt_j
    X = pi_xyz - ci; Y = pj_xyz - cj
    X[~mi] = 0; Y[~mj] = 0
    if has_color:
        Xc = torch.cat([X, pi[..., 3:6] * color_weight], -1)
        Yc = torch.cat([Y, pj[..., 3:6] * color_weight], -1)
    else:
        Xc, Yc = X, Y

    R = torch.eye(3, device=device).expand(B, 3, 3).clone()
    T = torch.zeros(B, 3, device=device)
    big = torch.finfo(torch.float32).max
    wi = mi.float()
    prev_err = None
    for it in range(max_iter):
        Xt = X @ R.transpose(1, 2) + T.unsqueeze(1)                       # (B, K, 3)
        Xq = torch.cat([Xt, Xc[..., 3:]], -1) if has_color else Xt
        d = torch.cdist(Xq, Yc)                                            # (B, Ki, Kj)
        d = d.masked_fill(~mj.unsqueeze(1), big)
        dmin, nn = d.min(dim=2)                                            # (B, Ki)
        Ynn = torch.gather(Y, 1, nn.unsqueeze(-1).expand(-1, -1, 3))
        err = (dmin * wi).sum(1) / wi.sum(1).clamp(min=1)
        # Kabsch between original X and matched Y (absolute, not incremental)
        cx = (X * wi.unsqueeze(-1)).sum(1, keepdim=True) / wi.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        cy = (Ynn * wi.unsqueeze(-1)).sum(1, keepdim=True) / wi.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        R = _kabsch(X - cx, Ynn - cy, wi)
        T = (cy - (R @ cx.transpose(1, 2)).transpose(1, 2)).squeeze(1)
        if prev_err is not None and torch.all((prev_err - err).abs() < 1e-6):
            break
        prev_err = err

    # back to original space: y = R (x - ci) + T + cj  =>  T_orig = cj - R ci + T
    ci_sq, cj_sq = ci.squeeze(1), cj.squeeze(1)
    T_orig = cj_sq - (R @ ci_sq.unsqueeze(-1)).squeeze(-1) + T
    bad = ~valid | torch.isnan(R).any(dim=(1, 2)) | torch.isnan(T_orig).any(dim=1)
    if bad.any():
        eye = torch.eye(3, device=device, dtype=R.dtype).expand(B, -1, -1)
        R = torch.where(bad[:, None, None], eye, R)
        T_orig = torch.where(bad[:, None], torch.zeros_like(T_orig), T_orig)
    return R, T_orig


def batch_icp_rotation(pts_i, pts_j, mask_i, mask_j, max_points=500, max_iter=50):
    R, _ = batch_icp(pts_i, pts_j, mask_i, mask_j, max_points=max_points, max_iter=max_iter)
    return R


def batch_centroid(pts, mask):
    xyz = pts[..., :3]
    mask = batch_outlier_removal(pts, mask)
    pts_z = xyz.clone(); pts_z[~mask] = 0
    cnt = mask.float().sum(1, keepdim=True).clamp(min=1)
    centroids = pts_z.sum(1) / cnt
    diff = xyz - centroids.unsqueeze(1); diff[~mask] = 0
    var = (diff ** 2 * mask.float().unsqueeze(-1)).sum(1) / cnt
    stds = var.sum(-1).sqrt()
    empty = mask.sum(1) == 0
    if empty.any():
        centroids = torch.where(empty[:, None], torch.zeros_like(centroids), centroids)
        stds = torch.where(empty, torch.zeros_like(stds), stds)
    return centroids, stds


def pad_pointclouds(pc_list, max_k=5000):
    N = len(pc_list); dim = 3
    for pc in pc_list:
        if len(pc) > 0:
            dim = pc.shape[-1]; break
    pts = torch.zeros(N, max_k, dim); mask = torch.zeros(N, max_k, dtype=torch.bool)
    for i, pc in enumerate(pc_list):
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(np.ascontiguousarray(pc))
        k = min(len(pc), max_k)
        if k > 0:
            pts[i, :k] = pc[:k].float(); mask[i, :k] = True
    return pts, mask


if __name__ == '__main__':
    # self-test: random cloud, known rotation
    torch.manual_seed(0)
    from scipy.spatial.transform import Rotation as Rot
    X = torch.randn(4, 800, 3) * 0.05
    Rg = torch.tensor(Rot.random(4, random_state=1).as_matrix(), dtype=torch.float32)
    # small rotations only (ICP basin)
    Rg = torch.tensor(Rot.from_rotvec(np.deg2rad(np.random.default_rng(0).uniform(-20, 20, (4, 3)))).as_matrix(), dtype=torch.float32)
    tg = torch.randn(4, 3) * 0.02
    Y = X @ Rg.transpose(1, 2) + tg[:, None]
    m = torch.ones(4, 800, dtype=torch.bool)
    R, T = batch_icp(X, Y, m, m)
    err = [np.degrees(np.linalg.norm(Rot.from_matrix(R[b].numpy() @ Rg[b].numpy().T).as_rotvec())) for b in range(4)]
    print('rot err deg', np.round(err, 3), 'trans err mm', ((T - tg).norm(dim=1) * 1000).numpy().round(2))
