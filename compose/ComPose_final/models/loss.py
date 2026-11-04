import torch
import torch.nn as nn
import torch.nn.functional as F


def rotmat_to_quat(R):
    """Rotation matrix (B, 3, 3) → unit quaternion (B, 4) [w, x, y, z].
    Numerically stable via Shepperd's method."""
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Four candidates
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = (trace + 1).clamp(min=1e-10).sqrt() * 2  # s = 4w
    m1 = trace > 0
    q[m1, 0] = s[m1] / 4
    q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s[m1]
    q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s[m1]
    q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s[m1]

    # Case 2: R[0,0] largest diagonal
    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = (1 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]).clamp(min=1e-10).sqrt() * 2
    q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s2
    q[m2, 1] = s2 / 4
    q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s2
    q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s2

    # Case 3: R[1,1] largest
    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = (1 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]).clamp(min=1e-10).sqrt() * 2
    q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s3
    q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s3
    q[m3, 2] = s3 / 4
    q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s3

    # Case 4: R[2,2] largest
    m4 = (~m1) & (~m2) & (~m3)
    s4 = (1 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]).clamp(min=1e-10).sqrt() * 2
    q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s4
    q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s4
    q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s4
    q[m4, 3] = s4 / 4

    return F.normalize(q, dim=-1)


def quat_to_rotmat(q):
    """Unit quaternion (B, 4) [w, x, y, z] → rotation matrix (B, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).view(-1, 3, 3)
    return R


def slerp(q0, q1, t):
    """Differentiable SLERP between quaternions q0, q1 with parameter t (B,).
    q0, q1: (B, 4), t: (B,) in [0, 1].
    t=0 → q0, t=1 → q1.
    """
    # Ensure shortest path
    dot = (q0 * q1).sum(-1, keepdim=True)  # (B, 1)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1 - 1e-6)

    theta = torch.acos(dot)  # (B, 1)
    t = t.view(-1, 1)  # (B, 1)

    sin_theta = theta.sin().clamp(min=1e-8)
    s0 = ((1 - t) * theta).sin() / sin_theta
    s1 = (t * theta).sin() / sin_theta

    return F.normalize(s0 * q0 + s1 * q1, dim=-1)


def rotation_slerp(R0, R1, alpha):
    """SLERP between rotation matrices. alpha=0 → R0, alpha=1 → R1.
    R0, R1: (B, 3, 3), alpha: (B,) in [0, 1].
    Returns: (B, 3, 3) valid rotation matrix.
    """
    q0 = rotmat_to_quat(R0)
    q1 = rotmat_to_quat(R1)
    q_blend = slerp(q0, q1, alpha)
    return quat_to_rotmat(q_blend)


class JointSelectionLoss(nn.Module):
    def __init__(self, trans_weight=10.0, bound_weight=1.0):
        super().__init__()
        self.trans_weight = trans_weight
        self.bound_weight = bound_weight

    def geodesic(self, R1, R2, eps=1e-6):
        R_rel = R1.transpose(-1, -2) @ R2
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps)
        return torch.acos(cos)

    def forward(self, R_hand_rel, R_obj_rel, alpha, gt_R, weights,
                pred_pos_i=None, pred_pos_j=None,
                gt_pos_i=None, gt_pos_j=None,
                intrinsics=None, pc_std_i=None, pc_std_j=None):
        B = gt_R.shape[0]

        # ── Rotation loss (SLERP blend) ──
        R_blend = rotation_slerp(R_obj_rel, R_hand_rel, alpha)
        rot_loss = self.geodesic(R_blend, gt_R).mean()

        # ── Translation loss: projection in pixel space ──
        trans_loss = torch.tensor(0.0, device=gt_R.device)
        bound_loss = torch.tensor(0.0, device=gt_R.device)
        trans_err_px = torch.tensor(0.0, device=gt_R.device)

        # Collect valid frames
        preds, gts, stds = [], [], []
        for pp, gp, ps in [(pred_pos_i, gt_pos_i, pc_std_i), (pred_pos_j, gt_pos_j, pc_std_j)]:
            if pp is not None and gp is not None:
                preds.append(pp); gts.append(gp); stds.append(ps)

        if preds and intrinsics is not None:
            pred_all = torch.stack(preds, dim=1)  # (B, N, 3)
            gt_all = torch.stack(gts, dim=1)      # (B, N, 3)
            N = pred_all.shape[1]

            fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
            fy = intrinsics[:, 1, 1].unsqueeze(1)
            cx = intrinsics[:, 0, 2].unsqueeze(1)
            cy = intrinsics[:, 1, 2].unsqueeze(1)
            # Image size from principal point (cx ≈ w/2, cy ≈ h/2)
            img_w = cx * 2  # (B, 1)
            img_h = cy * 2

            # Project to normalized coordinates [0, 1]
            pred_z = pred_all[..., 2].clamp(min=0.01)
            pred_ndc = torch.stack([
                (fx * pred_all[..., 0] / pred_z + cx) / img_w,
                (fy * pred_all[..., 1] / pred_z + cy) / img_h], dim=-1)
            gt_z = gt_all[..., 2].clamp(min=0.01)
            gt_ndc = torch.stack([
                (fx * gt_all[..., 0] / gt_z + cx) / img_w,
                (fy * gt_all[..., 1] / gt_z + cy) / img_h], dim=-1)

            trans_loss = (pred_ndc - gt_ndc).abs().sum(-1).mean()
            # Also compute pixel error for logging (using original resolution)
            # img_w/h: (B, 1) → broadcast to (B, N, 2)
            img_wh = torch.stack([img_w, img_h], dim=-1)  # (B, 1, 2)
            pred_uv = pred_ndc * img_wh
            gt_uv = gt_ndc * img_wh
            trans_err_px = (pred_uv - gt_uv).norm(dim=-1).mean()

            # Bound regularizer
            if stds[0] is not None:
                std_all = torch.stack(stds, dim=1)  # (B, N)
                deviation = (pred_all - gt_all).norm(dim=-1)  # (B, N)
                excess = F.relu(deviation - 0.5 * std_all)
                bound_loss = (excess ** 2).mean()

        total_loss = rot_loss + self.trans_weight * trans_loss + self.bound_weight * bound_loss

        with torch.no_grad():
            pred_err_deg = torch.rad2deg(self.geodesic(R_blend, gt_R)).mean()
            hand_err_deg = torch.rad2deg(self.geodesic(R_hand_rel, gt_R)).mean()
            obj_err_deg = torch.rad2deg(self.geodesic(R_obj_rel, gt_R)).mean()
            alpha_mean = alpha.mean()

        return total_loss, {
            'pred_err_deg': pred_err_deg,
            'hand_err_deg': hand_err_deg,
            'obj_err_deg': obj_err_deg,
            'alpha': alpha_mean,
            'rot_loss': rot_loss.detach(),
            'trans_loss': trans_loss.detach(),
            'trans_err_px': trans_err_px.detach(),
            'bound_loss': bound_loss.detach(),
        }
