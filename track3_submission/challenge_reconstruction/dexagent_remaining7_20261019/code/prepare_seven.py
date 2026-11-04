#!/usr/bin/env python3
"""Prepare the seven new HaWoR/Sharpa inputs without modifying supplied objects.

The established adapter is reused through its public preparation function. A
private input hierarchy supplies paths expected by that older adapter. Objects
retain their exact source poses and visual/collision geometry. The supplied
table-aligned world is preserved; its z=0 plane is not fit to object bottoms.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import least_squares
cv2.setNumThreads(1)

ROOT = Path("/data1/SRL_V2D")
BASE = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "challenge_reconstruction/dexagent_final_recon_20261017"
EPISODES = (14, 15, 16, 20, 25, 28, 30)
PUBLIC = Path("/data1/video_to_data_challenge/track_3")
sys.path.insert(0, str(LEGACY / "code"))
sys.path.insert(0, str(ROOT / "third_party/DexAgent"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def infer_table(first, K, world_from_camera, destination):
    """Fit the table edge across patches separated by the white dish rack.

    The old adapter selects one brown component; a dish rack divides the visible
    tabletop into several components. This uses their common outer envelope.
    The image-space gate excludes the brown cardboard box and hands at the sides.
    Only table geometry is inferred; no reference pose participates in the fit.
    """
    height, width = first.shape[:2]
    hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([4, 65, 35], np.uint8), np.array([35, 255, 255], np.uint8))
    mask[:int(.35 * height)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    selected = [index for index in range(1, count)
                if stats[index, cv2.CC_STAT_AREA] > .0035 * height * width
                and .30 * width < centroids[index, 0] < .85 * width
                and centroids[index, 1] > .40 * height]
    if not selected:
        raise ValueError("No image-supported tabletop components")
    keep = np.isin(labels, selected).astype(np.uint8)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.convexHull(np.concatenate(contours))[:, 0].astype(float)
    contour = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 3)))
                              for a, b in zip(contour, np.roll(contour, -1, axis=0))])
    contour = contour[(contour[:, 1] < height - 5) & (contour[:, 0] > 5)
                      & (contour[:, 0] < width - 5) & (contour[:, 1] > .35 * height + 5)]
    rays = (np.c_[contour, np.ones(len(contour))] @ np.linalg.inv(K).T) @ world_from_camera[:3, :3].T
    origin = world_from_camera[:3, 3]
    distance = -origin[2] / rays[:, 2]
    xy = (origin + rays * distance[:, None])[distance > 0, :2]
    if len(xy) < 100:
        raise ValueError("Too few calibrated tabletop edge samples")
    design = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    cx, cy, constant = np.linalg.lstsq(design, (xy * xy).sum(1), rcond=None)[0]
    initial = [cx, cy, np.sqrt(max(.001, constant + cx * cx + cy * cy))]
    fit = least_squares(lambda parameters: np.linalg.norm(xy - parameters[:2], axis=1) - parameters[2],
                        initial, loss="soft_l1", f_scale=.005)
    error = np.abs(np.linalg.norm(xy - fit.x[:2], axis=1) - fit.x[2])
    radius = float(fit.x[2])
    if not .15 < radius < .8 or float(np.median(error)) > .025:
        raise ValueError(f"Tabletop edge fit failed: radius={radius}, median residual={np.median(error)}")
    angle = np.linspace(0, 2 * np.pi, 361)
    points = np.c_[fit.x[0] + radius * np.cos(angle), fit.x[1] + radius * np.sin(angle), np.zeros(len(angle))]
    projected = ((points - origin) @ world_from_camera[:3, :3]) @ K.T
    projected = projected[:, :2] / projected[:, 2:]
    overlay = first.copy()
    cv2.polylines(overlay, [np.round(projected).astype(np.int32)], True, (0, 255, 0), 3)
    cv2.putText(overlay, f"Table edge: radius {radius:.3f} m, residual {1000*np.median(error):.1f} mm",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(destination), overlay)
    return dict(center_xy_m=fit.x[:2].tolist(), radius_m=radius, thickness_m=.04,
                method="Union of central tabletop color components; calibrated rays to supplied z=0 table; robust outer circle",
                selected_component_count=len(selected), edge_samples=len(xy),
                median_radial_residual_m=float(np.median(error)),
                inlier_fraction_15mm=float(np.mean(error < .015)), review_image=str(destination),
                limitation="Image-supported circular-table model; supplied table plane, 4cm thickness assumed")


def rectified_video(episode, camera, frames, fps):
    """Remap the real RGB to the exact ZIP pinhole camera, retaining all frames."""
    directory = BASE / "rgb" / episode
    output = directory / "anycalib/undistorted.mp4"
    report_path = directory / "rectification.json"
    calibration_path = PUBLIC / "public/meta/camera_calibration.json"
    calibration = json.loads(calibration_path.read_text())["cameras"]["ego_cam_a"]
    original = PUBLIC / "evaluation/videos/chunk-000/observation.images.ego_cam_a" / (episode + ".mp4")
    source_k = np.asarray(calibration["intrinsic_matrix"], float)
    target_k = np.asarray(camera["K"], float)
    width, height = map(int, calibration["resolution"])
    expected = dict(source=str(original), source_sha256=sha(original),
                    calibration_sha256=sha(calibration_path),
                    output_K=target_k.tolist(), width=width, height=height,
                    frames=int(frames), fps=float(fps))
    if report_path.exists() and output.exists():
        report = json.loads(report_path.read_text())
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing rectification belongs to different source/calibration")
        if sha(output) != report["output_sha256"]:
            raise ValueError("Rectified video changed")
        return directory, report
    if output.exists():
        raise FileExistsError(f"Unverified prior video: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(original))
    if not cap.isOpened():
        raise FileNotFoundError(original)
    if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
        raise ValueError("Public video dimensions do not match calibration")
    if abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-5:
        raise ValueError("Public video/reference frame rate mismatch")
    maps = cv2.initUndistortRectifyMap(source_k, np.asarray(calibration["distortion_coefficients"]),
                                     np.eye(3), target_k, (width, height), cv2.CV_32FC1)
    temporary = output.with_name("undistorted.partial.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Video writer unavailable")
    count = 0
    try:
        while True:
            ok, rgb = cap.read()
            if not ok:
                break
            writer.write(cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR))
            count += 1
    finally:
        writer.release()
        cap.release()
    if count != frames:
        raise ValueError(f"Public video has {count} frames, reference has {frames}")
    temporary.replace(output)
    report = dict(expected, output=str(output), output_sha256=sha(output),
                  method="OpenCV calibrated perspective undistortion; output intrinsics exactly camera_motion.npz K",
                  image_rotation="None: public MP4 already has display orientation; calibration metadata is not reapplied",
                  unchanged_original_video=True)
    write(report_path, report)
    return directory, report


def stage_source(episode, source_root):
    source = source_root / episode
    if not (source / "hawor_complete.json").is_file():
        raise FileNotFoundError("HaWoR conversion has not completed: " + str(source / "hawor_complete.json"))
    required = ["reference.parquet", "object_poses.npz", "camera_motion.npz", "config.json", "hand_report.json"]
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    hand_report = json.loads((source / "hand_report.json").read_text())
    if "hawor" not in json.dumps(hand_report).lower():
        raise ValueError("New source must explicitly document HaWoR hand provenance")
    row = pq.read_table(source / "reference.parquet").to_pylist()[0]
    frames, fps = len(row["object_body_position"]), float(row["fps"])
    with np.load(source / "camera_motion.npz", allow_pickle=False) as data:
        camera = {key: data[key].copy() for key in data.files}
    video_root, video_report = rectified_video(episode, camera, frames, fps)
    private = BASE / "adapter_sources" / episode
    if private.exists():
        raise FileExistsError("Preserve prior adapter input: " + str(private))
    # Copy small source inputs and meshes, excluding synthetic comparison videos.
    shutil.copytree(source, private, ignore=shutil.ignore_patterns("*.mp4", "__pycache__"))
    original_object_hash = sha(source / "object_poses.npz")
    asset_relocations = []
    for urdf in sorted((private / "assets").glob("*.urdf")):
        tree = ET.parse(urdf)
        changed = False
        for mesh in tree.getroot().iter("mesh"):
            old = mesh.get("filename", "")
            if Path(old).is_absolute():
                candidate = Path(old)
                parts = candidate.parts
                if "assets" in parts:
                    relative = Path(*parts[parts.index("assets") + 1:])
                else:
                    relative = Path(candidate.name)
                if not (private / "assets" / relative).is_file():
                    raise FileNotFoundError("Cannot relocate URDF mesh: " + old)
                mesh.set("filename", str(relative))
                asset_relocations.append(dict(urdf=urdf.name, original=old, relative=str(relative)))
                changed = True
        if changed:
            tree.write(urdf, encoding="utf-8", xml_declaration=True)
    config = json.loads((private / "config.json").read_text())
    config["source"] = str(video_root)
    config["input_adapter_note"] = "Video path added only; supplied table world and object trajectories retained"
    write(private / "config.json", config)
    names = np.load(private / "object_poses.npz", allow_pickle=False)["names"].tolist()
    manifest_path = private / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(episode=episode, frames=frames, fps=fps, objects=names)
    write(manifest_path, manifest)
    K = camera["K"]
    write(private / "stereo_calibration/rgb_intrinsics.json", dict(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=video_report["width"], height=video_report["height"],
        source="Exact ZIP camera_motion K; real RGB undistorted into this camera"))
    link = BASE / "input_root/challenge_reconstruction/final_recon" / episode
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(private, target_is_directory=True)
    if sha(private / "object_poses.npz") != original_object_hash:
        raise AssertionError("Source object poses changed")
    return private, dict(hawor_source=str(source), hawor_hand_report_sha256=sha(source / "hand_report.json"),
                         source_reference_sha256=sha(source / "reference.parquet"),
                         object_poses_sha256=original_object_hash,
                         camera_motion_sha256=sha(source / "camera_motion.npz"),
                         asset_path_relocations=asset_relocations, video=video_report)


def prepare_episode(number, source_root):
    if number not in EPISODES:
        raise ValueError("Only the seven new episodes are permitted")
    episode = f"episode_{number:06d}"
    destination = BASE / "episodes" / episode
    completion = destination / "preparation_complete.json"
    if completion.exists():
        return json.loads(completion.read_text())
    if destination.exists():
        raise FileExistsError("Preserve partial preparation: " + str(destination))
    private, evidence = stage_source(episode, source_root)
    import prepare_episodes as adapter
    from evaluate_mp_sr import geometry_from_reference
    adapter.ROOT = BASE / "input_root"
    adapter.WORKSPACE = BASE
    adapter.infer_table = infer_table
    # ROBOT and DEX stay the verified repository assets/upstream implementation.
    support = dict(method="Supplied reconstruction table-aligned world, table plane z=0",
                   source_config=str(private / "config.json"),
                   source_config_sha256=sha(private / "config.json"),
                   supplied_world=json.loads((private / "config.json").read_text()).get("world"),
                   no_pose_fit_or_object_bottom_shift=True)
    manifest = adapter.prepare(number, np.eye(4), support)
    manifest["support"]["method"] = support["method"]
    manifest["support"]["limitation"] = "Supplied reconstruction table plane; circle fitted from calibrated real RGB, 4cm thickness assumed"
    manifest["hand_source"] = "HaWoR human hand reconstruction retargeted to exact repository floating Sharpa hands"
    manifest["new_input_provenance"] = evidence
    write(destination / "input_manifest.json", manifest)
    write(destination / "metric_geometry.json", geometry_from_reference(private))
    with np.load(private / "object_poses.npz", allow_pickle=False) as obj, np.load(destination / "reference_trajectory.npz", allow_pickle=False) as ref:
        position_difference = float(np.max(np.abs(obj["world_from_object"][..., :3, 3] - ref["object_positions"])))
    if position_difference > 1e-12:
        raise AssertionError("Object translations changed during scene preparation")
    result = dict(episode=episode, input_dir=str(destination), source=str(private),
                  object_pose_bytes_unchanged=sha(private / "object_poses.npz") == evidence["object_poses_sha256"],
                  object_position_max_error_m=position_difference,
                  world_transform=np.eye(4).tolist(), provenance=evidence,
                  adapter_sha256=sha(LEGACY / "code/prepare_episodes.py"),
                  wrapper_sha256=sha(__file__), completed=True, physics_success_not_yet_tested=True)
    write(completion, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(EPISODES))
    parser.add_argument("--source-root", type=Path, default=BASE / "hawor_recon")
    args = parser.parse_args()
    for episode_number in args.episodes:
        prepare_episode(episode_number, args.source_root.resolve())
