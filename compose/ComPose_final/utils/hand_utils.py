"""
Hand pose utilities for JointSelector training.
Position-based joint frame construction from WiLoR 3D joints.
"""

import numpy as np

# WiLoR/MANO 21-joint layout:
# 0=Wrist
# 1=Index_MCP, 2=Index_PIP, 3=Index_DIP, 4=Index_Tip
# 5=Middle_MCP, 6=Middle_PIP, 7=Middle_DIP, 8=Middle_Tip
# 9=Ring_MCP, 10=Ring_PIP, 11=Ring_DIP, 12=Ring_Tip
# 13=Pinky_MCP, 14=Pinky_PIP, 15=Pinky_DIP, 16=Pinky_Tip
# 17=Thumb_CMC, 18=Thumb_MCP, 19=Thumb_IP, 20=Thumb_Tip

# Joint frame triplets: (center, point_a, point_b)
# Each triplet defines a local coordinate frame at a hand region.
# Names describe the physical meaning for grasping.
FRAME_TRIPLETS = [
    (0, 1, 5),     # 0: Palm (wrist, index_mcp, middle_mcp) = hand_frame baseline
    (0, 5, 9),     # 1: Palm-ring (wrist, middle_mcp, ring_mcp)
    (1, 0, 2),     # 2: Index base (index_mcp, wrist, index_pip)
    (5, 0, 6),     # 3: Middle base (middle_mcp, wrist, middle_pip)
    (9, 0, 10),    # 4: Ring base (ring_mcp, wrist, ring_pip)
    (13, 0, 14),   # 5: Pinky base (pinky_mcp, wrist, pinky_pip)
    (17, 0, 18),   # 6: Thumb base (thumb_cmc, wrist, thumb_mcp)
    (3, 2, 4),     # 7: Index tip (index_dip, index_pip, index_tip)
    (7, 6, 8),     # 8: Middle tip (middle_dip, middle_pip, middle_tip)
    (11, 10, 12),  # 9: Ring tip (ring_dip, ring_pip, ring_tip)
    (15, 14, 16),  # 10: Pinky tip (pinky_dip, pinky_pip, pinky_tip)
    (19, 18, 20),  # 11: Thumb tip (thumb_ip, thumb_mcp, thumb_tip)
]

FRAME_NAMES = [
    'Palm', 'Palm-ring',
    'Index_base', 'Middle_base', 'Ring_base', 'Pinky_base', 'Thumb_base',
    'Index_tip', 'Middle_tip', 'Ring_tip', 'Pinky_tip', 'Thumb_tip',
]

N_FRAMES = len(FRAME_TRIPLETS)  # 12


def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8) if n > 1e-8 else v


def build_frame(center, point_a, point_b):
    """Build orthonormal frame from 3 points.
    z = center→point_b, x = center→point_a (orthogonalized), y = cross(z,x)

    Args:
        center, point_a, point_b: (3,) np arrays
    Returns:
        R: (3, 3) rotation matrix (columns = x, y, z axes)
    """
    z = _normalize(point_b - center)
    x = point_a - center
    x = x - np.dot(x, z) * z  # orthogonalize
    x = _normalize(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # (3, 3)


def joints_to_frames(joints_3d):
    """Build local coordinate frames for each triplet from 21 joint positions.

    Args:
        joints_3d: (21, 3) np array
    Returns:
        frames: (N_FRAMES, 3, 3) rotation matrices
    """
    frames = np.zeros((N_FRAMES, 3, 3), dtype=np.float32)
    for i, (c, a, b) in enumerate(FRAME_TRIPLETS):
        frames[i] = build_frame(joints_3d[c], joints_3d[a], joints_3d[b])
    return frames


def relative_frame_rotations(joints_3d_i, joints_3d_j):
    """Compute relative rotation of each joint frame between two frames.

    Args:
        joints_3d_i: (21, 3) frame i joint positions
        joints_3d_j: (21, 3) frame j joint positions
    Returns:
        R_rel: (N_FRAMES, 3, 3) relative rotations per joint frame
    """
    F_i = joints_to_frames(joints_3d_i)
    F_j = joints_to_frames(joints_3d_j)
    return F_j @ F_i.transpose(0, 2, 1)


def joints_to_2d(joints_3d, intrinsics=None, img_h=None, img_w=None, triplet_indices=None):
    """Get 2D positions for each triplet's center joint.

    WiLoR joints_3d is wrist-relative (not camera space), so we normalize
    the x,y coordinates to [-1, 1] based on the joint cloud's own extent.

    Args:
        joints_3d: (21, 3) - wrist-relative 3D positions
        intrinsics, img_h, img_w: unused (kept for API compat)
        triplet_indices: list of triplets, default FRAME_TRIPLETS
    Returns:
        joints_2d: (N_FRAMES, 2) normalized [-1, 1]
        valid_mask: (N_FRAMES,) bool
    """
    if triplet_indices is None:
        triplet_indices = FRAME_TRIPLETS

    # Normalize joint x,y to [-1, 1] based on joint cloud extent
    xy = joints_3d[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = (xy_max - xy_min).clip(min=1e-6)
    xy_norm = (xy - xy_min) / xy_range * 2 - 1  # [-1, 1]

    n = len(triplet_indices)
    joints_2d = np.zeros((n, 2), np.float32)
    valid = np.ones(n, dtype=bool)

    for i, (c, a, b) in enumerate(triplet_indices):
        joints_2d[i] = xy_norm[c]

    return joints_2d, valid
