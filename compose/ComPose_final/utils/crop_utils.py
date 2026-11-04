"""
Object-centered crop utilities for JointSelector training.
Crops square region around object mask, resizes to fixed DINOv2-compatible size.
"""

import numpy as np
import cv2
import PIL.Image


def object_center_crop(image, object_mask, intrinsics, scale=3.0,
                       target_size=308, hand_mask=None, depthmap=None):
    """Crop square region centered on object mask, resize to target_size.

    Args:
        image: PIL.Image (W, H)
        object_mask: (H, W) uint8
        intrinsics: (3, 3) np.float32
        scale: bbox size multiplier (to include hand context)
        target_size: output size (must be multiple of 14 for DINOv2)
        hand_mask: (H, W) uint8 or None
        depthmap: (H, W) np.float32 or None

    Returns:
        image_crop: PIL.Image (target_size, target_size)
        K_crop: (3, 3) adjusted intrinsics
        object_mask_crop: (target_size, target_size) uint8
        hand_mask_crop: or None
        depthmap_crop: or None
        crop_box: (x1, y1, x2, y2) for reference
    """
    W_img, H_img = image.size

    # Object mask bbox center
    ys, xs = np.where(object_mask > 0)
    if len(ys) == 0:
        raise ValueError("Empty object mask")
    cx = int((xs.min() + xs.max()) / 2)
    cy = int((ys.min() + ys.max()) / 2)
    bbox_size = max(xs.max() - xs.min(), ys.max() - ys.min())

    # Square crop size from scale * bbox
    crop_size = max(int(bbox_size * scale), 28)

    # Crop box centered on object, clamped to image bounds
    half = crop_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Shift if out of bounds
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > W_img:
        x1 -= (x2 - W_img); x2 = W_img
    if y2 > H_img:
        y1 -= (y2 - H_img); y2 = H_img
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    actual_size = min(x2 - x1, y2 - y1)
    x2 = x1 + actual_size
    y2 = y1 + actual_size

    # Crop
    image_crop = image.crop((x1, y1, x2, y2))
    obj_crop = object_mask[y1:y2, x1:x2]

    # Resize to target_size
    resize_scale = target_size / actual_size
    image_crop = image_crop.resize((target_size, target_size), PIL.Image.LANCZOS)
    obj_crop = cv2.resize(obj_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # Adjust intrinsics: shift by crop origin, then scale
    K_crop = intrinsics.copy()
    K_crop[0, 2] -= x1
    K_crop[1, 2] -= y1
    K_crop[:2] *= resize_scale

    hm_crop = None
    if hand_mask is not None:
        hm_crop = cv2.resize(hand_mask[y1:y2, x1:x2],
                             (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    dm_crop = None
    if depthmap is not None:
        dm_crop = cv2.resize(depthmap[y1:y2, x1:x2],
                             (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return image_crop, K_crop, obj_crop, hm_crop, dm_crop, (x1, y1, x2, y2)
