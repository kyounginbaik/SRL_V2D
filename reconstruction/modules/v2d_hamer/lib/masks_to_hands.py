# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-track per-frame HaMeR regression driven by SAM2 hand masks.

For each (track, frame) pair, this module:
  1. Loads the SAM2 mask, computes its centroid and tight bbox.
  2. Builds a square crop centred at the centroid, sized as
     ``max(mask_bbox_side) * bbox_expansion``  → forgiving to partial masks.
  3. Runs HaMeR on the crop.
  4. Saves a JSON record with MANO params + camera fields.

Output layout:
    <output_dir>/<track_id>/<frame_id:06d>.json

JSON schema (per file, single detection per track per frame):
    {
      "track_id":   int,
      "is_right":   bool,
      "frame_idx":  int,
      "image_size": [W, H],
      "bbox":       {"x0": ..., "y0": ..., "x1": ..., "y1": ...},
      "mano":       {
         "betas":         [10 floats],
         "global_orient": [3 floats],     # axis-angle, converted from rotmat
         "hand_pose":     [45 floats]     # axis-angle, 15×3, converted
      },
      "camera":     {"cam_t": [tx, ty, tz], "focal_length": float}
    }

Frames where the mask is below ``mask_min_pixels`` (or absent) get no
output file — downstream code should treat missing as "hand not visible."

Usage:
    python -m v2d.hamer.lib.masks_to_hands \\
        --frames_dir /data/frames \\
        --masks_dir  /data/masks \\
        --tracks_path /data/hand_tracks.json \\
        --output_dir /data/hamer \\
        --weights_dir /data/weights/hamer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.hamer.lib._hamer import get_model, run_hamer, rotmat_to_axis_angle


def _mask_centroid_and_size(mask: np.ndarray) -> tuple[float, float, float, int]:
    """Return (cx, cy, side, n_pixels) of the mask's tight bbox."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = float(max(x1 - x0, y1 - y0))
    return cx, cy, side, int(xs.size)


def _box_iou(a: dict, b: dict) -> float:
    x0, y0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a["x1"] - a["x0"]) * max(0.0, a["y1"] - a["y0"])
    area_b = max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])
    return inter / max(area_a + area_b - inter, 1e-9)


def _select_detector_box(detections: list[dict], is_right: bool, mask_box: dict) -> dict | None:
    """Pick the same-handed detector box with greatest overlap with the mask."""
    candidates = [d for d in detections if bool(d.get("is_right")) == is_right and "bbox" in d]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (_box_iou(d["bbox"], mask_box), float(d.get("score", 0.0))))


def masks_to_hands(
    frames_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    weights_dir: str,
    bbox_expansion: float = 1.7,
    mask_min_pixels: int = 256,
    detections_dir: str | None = None,
    crop_source: str = "mask",
) -> None:
    if crop_source not in {"mask", "detector"}:
        raise ValueError(f"crop_source must be mask or detector, got {crop_source!r}")
    if crop_source == "detector" and not detections_dir:
        raise ValueError("crop_source=detector requires detections_dir")
    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    hand_tracks = [t for t in tracks if t.get("role", "hand") == "hand"]
    if not hand_tracks:
        raise RuntimeError(f"No hand tracks in {tracks_path}")

    print(f"Loading HaMeR from {weights_dir}...")
    model, cfg = get_model(weights_dir)

    for track in hand_tracks:
        oid       = int(track["object_id"])
        is_right  = bool(track["is_right"])
        track_dir = os.path.join(output_dir, str(oid))
        os.makedirs(track_dir, exist_ok=True)

        mask_files = sorted(glob.glob(os.path.join(masks_dir, str(oid), "*.png")))
        side = "right" if is_right else "left"
        print(f"\nTrack {oid} ({side}): {len(mask_files)} mask frames")

        n_written = n_skipped = 0
        for mask_path in tqdm(mask_files, desc=f"  track {oid}", ncols=80, unit="frame"):
            frame_idx = int(os.path.splitext(os.path.basename(mask_path))[0])
            frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
            if not os.path.exists(frame_path):
                # try .jpg
                frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                if not os.path.exists(frame_path):
                    continue
            out_path = os.path.join(track_dir, f"{frame_idx:06d}.json")
            if os.path.exists(out_path):
                continue   # skip already-done frames

            mask = np.asarray(Image.open(mask_path)) > 0
            cx, cy, mask_side, npx = _mask_centroid_and_size(mask)
            if npx < mask_min_pixels:
                n_skipped += 1
                continue

            ys, xs = np.nonzero(mask)
            mask_box = {"x0": float(xs.min()), "y0": float(ys.min()),
                        "x1": float(xs.max()), "y1": float(ys.max())}
            selected_box = mask_box
            selected_source = "mask"
            detector_score = None
            if crop_source == "detector":
                detection_path = os.path.join(detections_dir, f"{frame_idx:06d}.json")
                detections = []
                if os.path.exists(detection_path):
                    with open(detection_path) as f:
                        detections = json.load(f)
                selected = _select_detector_box(detections, is_right, mask_box)
                if selected is not None:
                    selected_box = selected["bbox"]
                    selected_source = "detector"
                    detector_score = float(selected.get("score", 0.0))

            image = np.asarray(Image.open(frame_path).convert("RGB"))
            H, W = image.shape[:2]
            cx = (float(selected_box["x0"]) + float(selected_box["x1"])) * 0.5
            cy = (float(selected_box["y0"]) + float(selected_box["y1"])) * 0.5
            selected_side = max(float(selected_box["x1"]) - float(selected_box["x0"]),
                                float(selected_box["y1"]) - float(selected_box["y0"]))
            size = selected_side * bbox_expansion
            # Clamp center so the crop stays a real bbox (HaMeR's crop helper
            # handles out-of-image padding internally, but keeping center sane
            # avoids silly edge cases).
            cx = float(np.clip(cx, 0.0, W - 1))
            cy = float(np.clip(cy, 0.0, H - 1))

            out = run_hamer(model, cfg, image, cx, cy, size, is_right=is_right)

            half = float(size) * 0.5
            record = {
                "track_id":   oid,
                "is_right":   is_right,
                "frame_idx":  frame_idx,
                "image_size": [int(W), int(H)],
                "bbox": {
                    "x0": float(cx - half),
                    "y0": float(cy - half),
                    "x1": float(cx + half),
                    "y1": float(cy + half),
                },
                "mano": {
                    "betas":         out["betas"].tolist(),
                    "global_orient": rotmat_to_axis_angle(out["global_orient"]).reshape(-1).tolist(),
                    "hand_pose":     rotmat_to_axis_angle(out["hand_pose"]).reshape(-1).tolist(),
                },
                "camera": {
                    "cam_t":        out["pred_cam_t_full"].tolist(),
                    "focal_length": out["scaled_focal_length"],
                },
                "crop": {
                    "source": selected_source,
                    "requested_source": crop_source,
                    "bbox_expansion": float(bbox_expansion),
                    "mask_bbox": mask_box,
                    "detector_score": detector_score,
                },
            }
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            n_written += 1

        print(f"  track {oid}: wrote {n_written}, skipped {n_skipped} (mask too small)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",  required=True)
    parser.add_argument("--masks_dir",   required=True,
                        help="Root masks dir; contains <track_id>/*.png subdirs.")
    parser.add_argument("--tracks_path", required=True,
                        help="hand_tracks.json (object_id → is_right mapping).")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True,
                        help="Dir containing _DATA/hamer_ckpts/.")
    parser.add_argument("--bbox_expansion", type=float, default=1.7)
    parser.add_argument("--mask_min_pixels", type=int, default=256)
    parser.add_argument("--detections_dir")
    parser.add_argument("--crop_source", choices=("mask", "detector"), default="mask")
    args = parser.parse_args()
    masks_to_hands(
        frames_dir       = args.frames_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        weights_dir      = args.weights_dir,
        bbox_expansion   = args.bbox_expansion,
        mask_min_pixels  = args.mask_min_pixels,
        detections_dir   = args.detections_dir,
        crop_source      = args.crop_source,
    )


if __name__ == "__main__":
    main()
