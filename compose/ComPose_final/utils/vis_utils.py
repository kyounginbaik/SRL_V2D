"""
Visualization utilities for JointSelector tensorboard logging.
"""

import numpy as np
import torch
import cv2

# WiLoR 21-joint skeleton connections
SKELETON_21 = [
    # Index
    (0,1),(1,2),(2,3),(3,4),
    # Middle
    (0,5),(5,6),(6,7),(7,8),
    # Ring
    (0,9),(9,10),(10,11),(11,12),
    # Pinky
    (0,13),(13,14),(14,15),(15,16),
    # Thumb
    (0,17),(17,18),(18,19),(19,20),
]

FINGER_COLORS_21 = {
    'Index': (0, 255, 0),
    'Middle': (255, 128, 0),
    'Ring': (0, 128, 255),
    'Pinky': (255, 0, 255),
    'Thumb': (0, 255, 255),
}

JOINT_FINGER_21 = (
    ['Wrist'] +
    ['Index']*4 + ['Middle']*4 + ['Ring']*4 + ['Pinky']*4 + ['Thumb']*4
)


def project_joints_to_image(joints_3d, bbox, img_h, img_w):
    """Map WiLoR wrist-relative joints_3d into image pixels using hand bbox.

    Args:
        joints_3d: (21, 3) wrist-relative
        bbox: (4,) x1,y1,x2,y2 in original image
        img_h, img_w: target image size (crop)
    Returns:
        uv: (21, 2) pixel coords in [0, img_w] x [0, img_h]
    """
    xy = joints_3d[:, :2]
    xy_min, xy_max = xy.min(0), xy.max(0)
    xy_range = (xy_max - xy_min).clip(min=1e-6)
    xy_norm = (xy - xy_min) / xy_range  # [0, 1]

    # Map to bbox in original image
    bx1, by1, bx2, by2 = bbox
    uv_orig = np.zeros((21, 2), np.float32)
    uv_orig[:, 0] = bx1 + xy_norm[:, 0] * (bx2 - bx1)
    uv_orig[:, 1] = by1 + xy_norm[:, 1] * (by2 - by1)

    # Scale to crop image (assumes crop covers full original for now)
    # This is approximate - crop may not cover the hand bbox
    uv = uv_orig.copy()
    uv[:, 0] = uv_orig[:, 0] / 640 * img_w  # assuming 640x480 original
    uv[:, 1] = uv_orig[:, 1] / 480 * img_h

    return uv


def draw_hand_skeleton(img_bgr, uv, weights=None):
    """Draw 21-joint hand skeleton with weight-proportional highlighting.

    Args:
        img_bgr: (H, W, 3) uint8
        uv: (21, 2) pixel coords
        weights: (21,) procrustes weights per joint, or None
    """
    vis = img_bgr.copy()

    # Draw bones (dim)
    for (a, b) in SKELETON_21:
        p1 = tuple(uv[a].astype(int))
        p2 = tuple(uv[b].astype(int))
        cv2.line(vis, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

    if weights is not None and len(weights) >= 21:
        w = weights[:21]
        w_max = w.max() + 1e-8

        # Draw joints: size + color proportional to weight
        for j in range(21):
            pt = tuple(uv[j].astype(int))
            w_norm = w[j] / w_max  # 0~1

            # Radius: 2 (low weight) to 12 (high weight)
            radius = int(2 + w_norm * 10)

            # Color: gray (low) → red (high)
            r = int(50 + w_norm * 205)
            g = int(50 * (1 - w_norm))
            b = int(50 * (1 - w_norm))
            color = (b, g, r)  # BGR

            cv2.circle(vis, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(vis, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)

            # Show weight value for top joints
            if w_norm > 0.3:
                cv2.putText(vis, f'{w[j]:.2f}', (pt[0] + radius + 2, pt[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Top 3 label
        top3 = w.argsort()[::-1][:3]
        JNAMES = ['Wri', 'IM', 'IP', 'ID', 'IT',
                   'MM', 'MP', 'MD', 'MT',
                   'RM', 'RP', 'RD', 'RT',
                   'PM', 'PP', 'PD', 'PT',
                   'TC', 'TM', 'TI', 'TT']
        top_str = ' '.join([f'{JNAMES[j]}:{w[j]:.2f}' for j in top3])
        cv2.putText(vis, top_str, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        # No weights: just draw joints with finger colors
        for j in range(21):
            pt = tuple(uv[j].astype(int))
            finger = JOINT_FINGER_21[j]
            color = FINGER_COLORS_21.get(finger, (200, 200, 200))
            cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)

    return vis


def make_trans_vis_grid(imgs, j2d_crop_px, pred_trans, gt_trans,
                        intrinsics, object_masks, max_samples=4):
    """Translation visualization: pred/gt projected arrows.

    Args:
        imgs: (B, 3, H, W) tensor [0,1]
        j2d_crop_px: (B, 21, 2) tensor - joint pixel coords in crop space
        pred_trans: (B, 3) tensor - predicted 3D translation delta
        gt_trans: (B, 3) tensor - GT 3D translation delta
        intrinsics: (B, 3, 3) tensor
        object_masks: (B, H, W) tensor
        max_samples: int
    Returns:
        (N, 3, H, W) tensor
    """
    B, _, H, W = imgs.shape
    n = min(B, max_samples)

    imgs_np = (imgs[:n].detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    j2d_np = j2d_crop_px[:n].detach().cpu().numpy()
    masks_np = object_masks[:n].detach().cpu().numpy()
    pred_t = pred_trans[:n].detach().cpu().numpy()
    gt_t = gt_trans[:n].detach().cpu().numpy()
    K = intrinsics[:n].detach().cpu().numpy()

    vis_list = []
    for i in range(n):
        img_bgr = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2BGR)

        # Projected arrows: pred (green) vs GT (red)
        vis_arrow = img_bgr.copy()
        mask_bool = masks_np[i] > 0.5
        overlay = vis_arrow.copy()
        overlay[mask_bool] = [255, 128, 0]
        cv2.addWeighted(overlay, 0.3, vis_arrow, 0.7, 0, vis_arrow)

        fx, fy = K[i, 0, 0], K[i, 1, 1]
        cx, cy = W // 2, H // 2  # image center as arrow origin

        # Project 3D delta to 2D pixel displacement
        scale = 50.0  # scale for visibility
        pred_dx = pred_t[i, 0] * fx * scale
        pred_dy = pred_t[i, 1] * fy * scale
        gt_dx = gt_t[i, 0] * fx * scale
        gt_dy = gt_t[i, 1] * fy * scale

        # GT arrow (red)
        gt_end = (int(cx + gt_dx), int(cy + gt_dy))
        cv2.arrowedLine(vis_arrow, (cx, cy), gt_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(vis_arrow, 'GT', (gt_end[0]+4, gt_end[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Pred arrow (green)
        pred_end = (int(cx + pred_dx), int(cy + pred_dy))
        cv2.arrowedLine(vis_arrow, (cx, cy), pred_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(vis_arrow, 'Pred', (pred_end[0]+4, pred_end[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # Error text
        err_mm = np.linalg.norm(pred_t[i] - gt_t[i]) * 1000
        cv2.putText(vis_arrow, f'{err_mm:.0f}mm', (5, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        vis_list.append(torch.from_numpy(cv2.cvtColor(vis_arrow, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255)

    return torch.stack(vis_list)


def make_vis_grid(imgs, j2d_crop_px, weights, object_masks, max_samples=4):
    """Create tensorboard visualization: object mask + hand skeleton + selection.

    Args:
        imgs: (B, 3, H, W) tensor [0,1]
        j2d_crop_px: (B, 21, 2) tensor - joint pixel coords in crop space
        weights: (B, N) tensor
        object_masks: (B, H, W) tensor
        max_samples: int
    Returns:
        (N, 3, H, W) tensor
    """
    B, _, H, W = imgs.shape
    n = min(B, max_samples)

    imgs_np = (imgs[:n].detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    j2d_np = j2d_crop_px[:n].detach().cpu().numpy()
    w_np = weights[:n].detach().cpu().numpy()
    masks_np = object_masks[:n].detach().cpu().numpy()

    vis_list = []
    for i in range(n):
        img_bgr = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2BGR)

        # Object mask overlay
        vis_mask = img_bgr.copy()
        mask_bool = masks_np[i] > 0.5
        overlay = vis_mask.copy()
        overlay[mask_bool] = [255, 128, 0]
        cv2.addWeighted(overlay, 0.3, vis_mask, 0.7, 0, vis_mask)

        # Hand skeleton in crop space
        vis_hand = draw_hand_skeleton(img_bgr, j2d_np[i], w_np[i])

        vis_list.append(torch.from_numpy(cv2.cvtColor(vis_mask, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255)
        vis_list.append(torch.from_numpy(cv2.cvtColor(vis_hand, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255)

    return torch.stack(vis_list)
