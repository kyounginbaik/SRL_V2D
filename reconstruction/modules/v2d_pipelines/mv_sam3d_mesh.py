# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GT-free SAM2 view selection and MV-SAM3D mesh generation for ego reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from v2d.common.datatypes import DepthImage


# The same quality proxies and thresholds used in the public pot experiment.
# Pixel area is scaled from its 1920x1440 input; all other thresholds are fixed.
QUALITY_GATES = {
    "min_mask_pixels_at_1920x1440": 20000,
    "min_border_margin_px": 8,
    "min_largest_component_fraction": 0.95,
    "min_boundary_laplacian_variance": 80.0,
    "min_context_laplacian_variance": 50.0,
    "max_hand_proximity_fraction": 0.10,
    "min_valid_depth_fraction": 0.95,
    "min_depth_m": 0.15,
    "max_depth_m": 2.0,
    "max_depth_deviation_from_mask_median_m": 0.25,
    "min_near_median_depth_fraction": 0.95,
}
MAX_CANDIDATES = 48
MAX_VIEWS = 4
MV_SAM3D_COMMIT = "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform(values: list[int], count: int) -> list[int]:
    """Evenly sample ordered ranks, retaining both ends when count > 1."""
    if len(values) <= count:
        return values.copy()
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def record_single_view(*, output_dir: str, image_path: str, mask_path: str,
                       depth_path: str, object_id: int, object_prompt: str,
                       seed: int) -> dict:
    """Record the explicit legacy single-view choice without changing its behavior."""
    image, mask, depth = map(Path, (image_path, mask_path, depth_path))
    with Image.open(image) as rgb, Image.open(mask) as object_mask, Image.open(depth) as metric:
        if rgb.size != object_mask.size or rgb.size != metric.size:
            raise ValueError("Single-view RGB/object-mask/depth coordinate mismatch")
        if not np.asarray(object_mask.convert("L")).any():
            raise ValueError("Single-view object mask is empty")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"requested_backend": "sam3d", "actual_backend": "sam3d",
                "fallback_reason": None, "selector": "explicit_reference_frame",
                "selected_frames": [int(image.stem)], "object_id": object_id,
                "object_prompt": object_prompt, "mask_source": "SAM2 propagated object track",
                "seed": seed, "generation": {"script": "v2d.sam3d.docker.run_image_to_mesh"},
                "views": [{"frame": int(image.stem), "rgb_source": str(image),
                           "rgb_sha256": _digest(image), "mask_source": str(mask),
                           "mask_sha256": _digest(mask), "depth_source": str(depth)}]}
    (out / "mesh_generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def inspect_view(image_path: Path, mask_path: Path, depth_path: Path,
                 hand_mask_paths: list[Path] | None = None) -> dict:
    """Apply the experiment's mask, blur, occlusion and metric-depth gates."""
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_u8 is None:
        raise FileNotFoundError(f"Missing RGB/mask pair: {image_path}, {mask_path}")
    if frame.shape[:2] != mask_u8.shape:
        raise ValueError(f"RGB/mask coordinate mismatch: {image_path}, {mask_path}")
    with Image.open(depth_path) as depth_png:
        depth_m = DepthImage.from_pil_image(depth_png).depth
    if depth_m.shape != mask_u8.shape:
        raise ValueError(f"Depth/mask coordinate mismatch: {depth_path}, {mask_path}")
    mask = mask_u8 > 0
    area = int(mask.sum())
    info = {"frame": int(image_path.stem), "image": str(image_path),
            "mask": str(mask_path), "depth": str(depth_path),
            "mask_pixels": area, "rejection_reasons": []}
    if area == 0:
        info["rejection_reasons"].append("empty_object_mask")
        info["passes_quality"] = False
        return info
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    h, w = mask.shape
    margin = min(bbox[0], bbox[1], w - 1 - bbox[2], h - 1 - bbox[3])
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    largest = float(stats[1:, cv2.CC_STAT_AREA].max() / area) if count > 1 else 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.uint8)
    boundary = ((cv2.dilate(mask.astype(np.uint8), kernel) > 0) &
                (cv2.erode(mask.astype(np.uint8), kernel) == 0))
    boundary_var = float(cv2.Laplacian(gray, cv2.CV_64F)[boundary].var())
    pad = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.15)
    context = gray[max(0, bbox[1] - pad):min(h, bbox[3] + pad),
                   max(0, bbox[0] - pad):min(w, bbox[2] + pad)]
    context_var = float(cv2.Laplacian(context, cv2.CV_64F).var())
    hand = np.zeros_like(mask)
    used_hand_masks = []
    for path in hand_mask_paths or []:
        if not path.is_file():
            continue
        hand_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if hand_u8 is None or hand_u8.shape != mask.shape:
            raise ValueError(f"Hand/object mask coordinate mismatch: {path}")
        hand |= hand_u8 > 0
        used_hand_masks.append(str(path))
    hand_near = cv2.dilate(hand.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    hand_fraction = float(np.count_nonzero(mask & hand_near) / area)
    valid = (np.isfinite(depth_m) & (depth_m >= QUALITY_GATES["min_depth_m"]) &
             (depth_m <= QUALITY_GATES["max_depth_m"]))
    valid_fraction = float(np.count_nonzero(mask & valid) / area)
    values = depth_m[mask & valid]
    median = float(np.median(values)) if len(values) else float("nan")
    near = valid & (np.abs(depth_m - median) <=
                    QUALITY_GATES["max_depth_deviation_from_mask_median_m"])
    near_fraction = float(np.count_nonzero(mask & near) / max(1, len(values)))
    minimum_area = max(1, int(round(QUALITY_GATES["min_mask_pixels_at_1920x1440"]
                                    * w * h / (1920 * 1440))))
    checks = (
        (area < minimum_area, "mask_too_small"),
        (margin < QUALITY_GATES["min_border_margin_px"], "touches_frame_border"),
        (largest < QUALITY_GATES["min_largest_component_fraction"], "fragmented_or_leaking_mask"),
        (boundary_var < QUALITY_GATES["min_boundary_laplacian_variance"], "blur_proxy_below_threshold"),
        (context_var < QUALITY_GATES["min_context_laplacian_variance"], "context_motion_blur"),
        (hand_fraction > QUALITY_GATES["max_hand_proximity_fraction"], "hand_near_object_mask"),
        (valid_fraction < QUALITY_GATES["min_valid_depth_fraction"], "invalid_or_out_of_range_depth"),
        (near_fraction < QUALITY_GATES["min_near_median_depth_fraction"], "depth_background_leakage"),
    )
    info.update(bbox_xyxy=bbox, border_margin_px=int(margin),
                largest_component_fraction=largest,
                boundary_laplacian_variance=boundary_var,
                context_laplacian_variance=context_var,
                hand_proximity_fraction=hand_fraction,
                hand_masks=used_hand_masks, valid_depth_fraction=valid_fraction,
                median_depth_m=median if np.isfinite(median) else None,
                near_median_depth_fraction=near_fraction,
                min_mask_pixels_scaled=minimum_area)
    info["rejection_reasons"] = [reason for failed, reason in checks if failed]
    info["passes_quality"] = not info["rejection_reasons"]
    return info


def prepare_views(*, frames_dir: str, masks_dir: str, depth_dir: str,
                  object_id: int, reference_frame: int, output_dir: str,
                  seed: int, object_prompt: str) -> dict:
    """Select up to four GT-free views from the one SAM2 object track."""
    frames, masks, depths, out = map(Path, (frames_dir, masks_dir, depth_dir, output_dir))
    object_mask_dir = masks / str(object_id)
    if not object_mask_dir.is_dir():
        raise FileNotFoundError(f"Missing SAM2 object track: {object_mask_dir}")
    track_path = out.parent / "object_track.json"
    if not track_path.is_file():
        raise FileNotFoundError(f"Object track metadata missing: {track_path}")
    track = json.loads(track_path.read_text())
    if (track.get("object_id") != object_id or track.get("prompt") != object_prompt
            or track.get("reference_frame") != reference_frame):
        raise ValueError("Object track ID/prompt/reference frame differs from the requested object")
    frame_ids = sorted(int(p.stem) for p in frames.glob("*.png") if p.stem.isdecimal()
                       and (object_mask_dir / p.name).is_file()
                       and (depths / p.name).is_file())
    if reference_frame not in frame_ids:
        raise ValueError(f"Reference RGB/SAM2 mask/depth missing at {reference_frame}")
    dino_path = out.parent / "dino_detections.json"
    detector_check = None
    if dino_path.is_file():
        detections = json.loads(dino_path.read_text())
        if not detections or "box" not in detections[0]:
            raise ValueError(f"No reference object detection in {dino_path}")
        box = detections[0]["box"]
        with Image.open(object_mask_dir / f"{reference_frame:06d}.png") as image:
            ref_mask = np.asarray(image.convert("L")) > 0
        x0, y0 = max(0, int(box["x0"])), max(0, int(box["y0"]))
        x1, y1 = min(ref_mask.shape[1], int(np.ceil(box["x1"]))), min(ref_mask.shape[0], int(np.ceil(box["y1"])))
        inside = int(ref_mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
        fraction = inside / max(1, int(ref_mask.sum()))
        detector_check = {"source": str(dino_path), "reference_mask_in_detection_fraction": fraction}
        if fraction < 0.5:
            raise ValueError("SAM2 reference object mask does not match the detected object box")
    candidates = _uniform(frame_ids, MAX_CANDIDATES)
    if reference_frame not in candidates:
        candidates = sorted(set(candidates + [reference_frame]))
    hand_dirs = [p for p in masks.iterdir() if p.is_dir() and p.name.isdecimal()
                 and p.name != str(object_id)]
    records = []
    for index in candidates:
        stem = f"{index:06d}.png"
        records.append(inspect_view(frames / stem, object_mask_dir / stem,
                                    depths / stem, [p / stem for p in hand_dirs]))
    eligible = [r["frame"] for r in records if r["passes_quality"]]
    if reference_frame in eligible:
        others = [f for f in eligible if f != reference_frame]
        selected = [reference_frame] + _uniform(others, MAX_VIEWS - 1)
    else:
        selected = _uniform(eligible, MAX_VIEWS)
    if not selected:
        raise RuntimeError("No quality-passing RGB/object-mask/depth views")
    backend = "mvsam3d" if len(selected) >= 2 else "sam3d"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "requested_backend": "mvsam3d", "actual_backend": backend,
        "fallback_reason": "only_one_quality_passing_view" if backend == "sam3d" else None,
        "selector": "quality_gates_then_temporal_rank_uniform_max_4",
        "quality_gates": QUALITY_GATES, "candidate_limit": MAX_CANDIDATES,
        "reference_frame": reference_frame, "selected_frames": selected,
        "object_prompt": object_prompt, "object_id": object_id,
        "mask_source": "SAM2 propagated object track",
        "detector_identity_check": detector_check,
        "identity_limit": "Same SAM2 track ID and reference detector box do not prove that the track never switches objects",
        "seed": seed, "generation": {"script": "run_inference_weighted.py",
            "weighting": "official defaults, no DA3", "model_tag": "hf",
            "mv_sam3d_commit": MV_SAM3D_COMMIT},
        "candidate_records": records, "views": [],
    }
    input_root = out / "mv_input"
    image_dir, alpha_dir = input_root / "images", input_root / "object"
    image_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    for view_index, frame_id in enumerate(selected):
        stem = f"{frame_id:06d}.png"
        rgb_path = frames / stem
        mask_path = object_mask_dir / stem
        with Image.open(rgb_path) as im, Image.open(mask_path) as mm:
            rgb = im.convert("RGB")
            mask = np.asarray(mm.convert("L")) > 0
            if rgb.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"RGB/object mask shape mismatch at frame {frame_id}")
            rgba = rgb.copy()
            rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
            rgb.save(image_dir / f"{view_index}.png")
            rgba.save(alpha_dir / f"{view_index}.png")
        # Verify exactly the RGB and object mask the model loader will see.
        with Image.open(image_dir / f"{view_index}.png") as saved_rgb, \
             Image.open(alpha_dir / f"{view_index}.png") as saved_rgba:
            rgba_pixels = np.asarray(saved_rgba)
            if not np.array_equal(np.asarray(saved_rgb), rgba_pixels[..., :3]):
                raise RuntimeError("MV input RGB differs from its mask PNG RGB")
            if not np.array_equal(mask, rgba_pixels[..., 3] > 0):
                raise RuntimeError("MV input alpha differs from SAM2 object mask")
        manifest["views"].append({"view_index": view_index, "frame": frame_id,
                                  "rgb_source": str(rgb_path), "rgb_sha256": _digest(rgb_path),
                                  "mask_source": str(mask_path), "mask_sha256": _digest(mask_path)})
    manifest_path = out / "mesh_generation_manifest.json"
    manifest["input_root"] = str(input_root)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def run_mv_sam3d(*, manifest: dict, repo_dir: str, weights_dir: str,
                 docker_image: str, cache_dir: str | None = None,
                 lib_dir: str | None = None) -> Path:
    """Run the pinned official script and preserve its raw GLB plus an OBJ copy."""
    if manifest["actual_backend"] != "mvsam3d":
        raise ValueError("MV-SAM3D requires at least two selected views")
    repo = Path(repo_dir).resolve()
    weights = Path(weights_dir).resolve()
    input_root = Path(manifest["input_root"]).resolve()
    work = Path(manifest["manifest_path"]).parent.resolve()
    raw_glb, raw_obj = work / "raw_mv_sam3d.glb", work / "raw_mv_sam3d.obj"
    actual_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != MV_SAM3D_COMMIT:
        raise RuntimeError(f"MV-SAM3D checkout {actual_commit} != pinned {MV_SAM3D_COMMIT}")
    if not (weights / "pipeline.yaml").is_file():
        raise FileNotFoundError(f"Missing SAM3D checkpoint pipeline.yaml: {weights}")
    revision_file = weights.parent / ".cache/huggingface/download/checkpoints/pipeline.yaml.metadata"
    revision = revision_file.read_text().splitlines()[0] if revision_file.is_file() else None
    generation = manifest["generation"]
    generation.update({"mv_sam3d_commit": actual_commit,
                       "sam3d_weights_repo": "facebook/sam-3d-objects",
                       "sam3d_weights_revision": revision,
                       "checkpoint_config_sha256": _digest(weights / "pipeline.yaml"),
                       "docker_image": docker_image})
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")
    fingerprint = hashlib.sha256(json.dumps({
        "seed": manifest["seed"], "views": manifest["views"],
        "generation": generation}, sort_keys=True).encode()).hexdigest()
    source_record = work / "raw_mv_sam3d.source.json"
    if raw_glb.is_file() and raw_obj.is_file() and source_record.is_file():
        if json.loads(source_record.read_text()).get("fingerprint") == fingerprint:
            return raw_obj
    run_dir = work / "mv_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # The SAM3D downloader puts MoGe and DINO caches beside hf-download/.
    cache = Path(cache_dir).resolve() if cache_dir else weights.parent.parent
    cache.mkdir(parents=True, exist_ok=True)
    (run_dir / "tmp").mkdir(exist_ok=True)
    command = ["docker", "run", "--rm", "--gpus",
               f"device={os.environ['SLURM_JOB_GPUS']}" if os.environ.get("SLURM_JOB_GPUS") else "all",
               "--shm-size=8g", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "HOME=/data/cache", "-e", "HF_HOME=/data/cache/hf_home",
               "-e", "TORCH_HOME=/data/cache/torch_home", "-e", "XDG_CACHE_HOME=/data/cache/xdg",
               "-e", "TMPDIR=/data/output/tmp", "-e", "MVSAM3D_OUTPUT_ROOT=/data/output",
               "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", f"{repo}:/mv:ro", "-v", f"{weights}:/mv/checkpoints/hf:ro",
               "-v", f"{input_root}:/data/input:ro", "-v", f"{run_dir}:/data/output:rw",
               "-v", f"{cache}:/data/cache:rw"]
    if lib_dir:
        library = Path(lib_dir).resolve()
        command += ["-v", f"{library}:/data/lib:ro", "-e",
                    "LD_LIBRARY_PATH=/data/lib:/opt/conda/lib:/usr/local/cuda/lib64"]
    command += ["-w", "/mv", docker_image, "python", "run_inference_weighted.py",
                "--input_path", "/data/input", "--mask_prompt", "object",
                "--image_names", ",".join(str(i) for i in range(len(manifest["selected_frames"]))),
                "--model_tag", "hf", "--seed", str(manifest["seed"])]
    log_path = work / "mv_sam3d.log"
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    matches = [p for p in run_dir.rglob("result.glb") if p.stat().st_size > 0]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one MV-SAM3D result.glb, found {len(matches)}; see {log_path}")
    preserve_raw_mesh(matches[0], work)
    source_record.write_text(json.dumps({"fingerprint": fingerprint,
                                         "raw_glb_sha256": _digest(raw_glb),
                                         "raw_obj_sha256": _digest(raw_obj)}, indent=2) + "\n")
    return raw_obj


def preserve_raw_mesh(result_glb: Path, output_dir: Path) -> Path:
    """Keep unaligned model output and convert its own coordinates to OBJ for FP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_glb, raw_obj = output_dir / "raw_mv_sam3d.glb", output_dir / "raw_mv_sam3d.obj"
    if result_glb.resolve() != raw_glb.resolve():
        shutil.copy2(result_glb, raw_glb)
    mesh = trimesh.load(raw_glb, force="mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"MV-SAM3D returned an empty mesh: {raw_glb}")
    mesh.export(raw_obj)
    return raw_obj
