#!/usr/bin/env python3
"""Track 3 MP-SR against reconstructed reference poses, NOT confidential GT.

Official definitions audited against the submission kit downloaded 2026-10-17.
NPZ schema and limitations are documented in ../METRIC.md. No temporal warping,
per-object trajectory alignment, missing-object omission, or partial-horizon score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

KIT_URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip"
KIT_SHA256 = "8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e"
SYMMETRIES = {"beige_cup": "continuous_axial", "blue_cup": "continuous_axial", "white_pot_lid": "half_turn"}
POSITION_THRESHOLD_M = 0.03
ORIENTATION_THRESHOLD_RAD = np.deg2rad(30.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_poses(data: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize to T,W,O with normalized scalar-first quaternions."""
    times = np.asarray(data["timestamps_seconds"], dtype=np.float64)
    positions = np.asarray(data["object_positions"], dtype=np.float64)
    quaternions = np.asarray(data["object_quaternions_wxyz"], dtype=np.float64)
    names = [str(x) for x in data["object_names"]]
    if positions.ndim == 3:
        positions = positions[:, None]
        quaternions = quaternions[:, None]
    if positions.ndim != 4 or positions.shape[-1] != 3 or quaternions.shape != (*positions.shape[:-1], 4):
        raise ValueError("Require positions[T,(W),O,3], quaternions_wxyz[T,(W),O,4]")
    if times.ndim != 1 or len(times) != len(positions) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Require at least two strictly increasing explicit timestamps matching pose frames")
    if len(names) != positions.shape[2] or len(set(names)) != len(names) or not names or positions.shape[1] < 1:
        raise ValueError("Missing/duplicate object names or empty world/object axis")
    if not all(np.isfinite(x).all() for x in (times, positions, quaternions)):
        raise ValueError("Non-finite poses/timestamps cannot be scored")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(abs(norms - 1.0) > 1e-3):
        raise ValueError("Non-unit or zero quaternion; expected wxyz with norm within 0.001 of 1")
    return dict(timestamps_seconds=times, object_positions=positions.copy(),
                object_quaternions_wxyz=quaternions / norms, object_names=names)


def load_trajectory(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        required = ("timestamps_seconds", "object_positions", "object_quaternions_wxyz", "object_names")
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError(f"Missing recording fields: {sorted(missing)}")
        return validate_poses({key: source[key] for key in required})


def load_reference(reference_dir: str | Path) -> dict[str, Any]:
    directory = Path(reference_dir)
    if directory.is_file():
        result = load_trajectory(directory)
    else:
        meta = json.loads((directory / "manifest.json").read_text())
        fps = float(meta["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid reference fps")
        with np.load(directory / "object_poses.npz", allow_pickle=False) as source:
            poses = np.asarray(source["world_from_object"], float)
            names = source["names"].copy()
            if poses.ndim != 4 or poses.shape[1:] != (len(names), 4, 4):
                raise ValueError("Invalid reference world_from_object matrices")
            if "frames" in source and not np.array_equal(source["frames"], np.arange(len(poses))):
                raise ValueError("Reference must include every original frame, beginning at zero")
        rotations = poses[..., :3, :3]
        if (not np.isfinite(poses).all() or not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5)
                or np.any(np.linalg.det(rotations) < .999)):
            raise ValueError("Reference poses must be rigid transforms")
        if int(meta["frames"]) != len(poses):
            raise ValueError("Reference frame count disagrees with manifest")
        q = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(*poses.shape[:2], 4)
        result = validate_poses(dict(timestamps_seconds=np.arange(len(poses)) / fps,
                                    object_positions=poses[..., :3, 3],
                                    object_quaternions_wxyz=q[..., [3, 0, 1, 2]], object_names=names))
    if result["object_positions"].shape[1] != 1:
        raise ValueError("Reference must contain exactly one world")
    return result


def rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternions_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(
        *quaternions_wxyz.shape[:-1], 3, 3)


def apply_world_transform(data: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    transform = np.asarray(transform, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)):
        raise ValueError("World transform must be one rigid SE(3) transform; scaling is prohibited")
    result = dict(data)
    result["object_positions"] = data["object_positions"] @ transform[:3, :3].T + transform[:3, 3]
    rotations = transform[:3, :3] @ rotation_matrices(data["object_quaternions_wxyz"])
    result["object_quaternions_wxyz"] = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(
        *rotations.shape[:-2], 4)[..., [3, 0, 1, 2]]
    return result


def resample_to_reference(trajectory: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Sample original absolute times; interpolation is bounded to <=1.5 reference intervals."""
    times, target = trajectory["timestamps_seconds"], reference["timestamps_seconds"]
    if set(trajectory["object_names"]) != set(reference["object_names"]):
        raise ValueError("Recording must contain exactly every reference object; no missing/extraneous objects")
    tolerance = 1e-6
    if times[0] > target[0] + tolerance or times[-1] < target[-1] - tolerance:
        raise ValueError(f"Incomplete horizon: recording [{times[0]}, {times[-1]}], reference [{target[0]}, {target[-1]}]")
    intervals = np.diff(times)
    overlaps = (times[:-1] < target[-1]) & (times[1:] > target[0])
    if np.any(intervals[overlaps] > 1.5 * np.median(np.diff(target)) + tolerance):
        raise ValueError("Missing temporal coverage: recording gaps exceed 1.5 reference frame intervals")
    order = [trajectory["object_names"].index(name) for name in reference["object_names"]]
    positions = trajectory["object_positions"][:, :, order]
    quaternions = trajectory["object_quaternions_wxyz"][:, :, order]
    output_pos = np.empty((len(target), *positions.shape[1:]))
    output_quat = np.empty((len(target), *quaternions.shape[1:]))
    sample_times = np.clip(target, times[0], times[-1])  # at most rounding tolerance at endpoints
    for world in range(positions.shape[1]):
        for body in range(positions.shape[2]):
            for axis in range(3):
                output_pos[:, world, body, axis] = np.interp(sample_times, times, positions[:, world, body, axis])
            rotations = Rotation.from_quat(quaternions[:, world, body, :][:, [1, 2, 3, 0]])
            output_quat[:, world, body] = Slerp(times, rotations)(sample_times).as_quat()[:, [3, 0, 1, 2]]
    return dict(timestamps_seconds=target, object_positions=output_pos,
                object_quaternions_wxyz=output_quat, object_names=reference["object_names"])


def geometry_from_reference(reference_dir: str | Path) -> dict[str, Any]:
    """Declared physical symmetry classes; local proxy axes from exact source meshes."""
    import trimesh
    directory = Path(reference_dir)
    names = load_reference(directory)["object_names"]
    objects = {}
    for name in names:
        mesh_path = next((directory / "assets" / f"{name}{suffix}" for suffix in (".obj", ".ply", ".glb")
                          if (directory / "assets" / f"{name}{suffix}").exists()), None)
        if mesh_path is None:
            raise ValueError(f"Missing exact reference mesh for {name}")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
            raise ValueError(f"Invalid geometry for {name}")
        item = dict(symmetry=SYMMETRIES.get(name, "none"),
                    anchor_local_m=np.asarray(mesh.centroid).tolist(),
                    anchor_source="surface centroid of exact local reconstruction mesh, not confidential scan",
                    mesh_path=str(mesh_path.resolve()), mesh_sha256=sha256(mesh_path), axis_local=[0., 0., 0.],
                    axis_source="not applicable: official full orientation")
        if item["symmetry"] != "none":
            surface, _ = trimesh.sample.sample_surface(mesh, 50000, seed=0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(surface.T))
            index = 0 if item["symmetry"] == "half_turn" else 2
            gap = eigenvalues[1] / max(eigenvalues[0], 1e-12) if index == 0 else eigenvalues[2] / max(eigenvalues[1], 1e-12)
            if gap < 1.5:
                raise ValueError(f"Ambiguous proxy symmetry axis for {name}; supply reviewed geometry JSON")
            axis = eigenvectors[:, index]
            axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
            item.update(axis_local=axis.tolist(), axis_source="local mesh surface covariance, minimum axis for lid / maximum axis for tall cups; estimated proxy",
                        axis_eigenvalues_m2=eigenvalues.tolist(), axis_eigenvalue_gap=float(gap))
        objects[name] = item
    return dict(official_symmetry_source=KIT_URL, official_kit_sha256=KIT_SHA256,
                confidential_geometry_available=False, objects=objects)


def validate_geometry(names: list[str], geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    anchors, axes, kinds = [], [], []
    for name in names:
        entry = geometry["objects"][name]
        kind = entry["symmetry"]
        if kind != SYMMETRIES.get(name, "none"):
            raise ValueError(f"Undeclared object symmetry for {name}: {kind}")
        anchor, axis = np.asarray(entry["anchor_local_m"], float), np.asarray(entry["axis_local"], float)
        if anchor.shape != (3,) or axis.shape != (3,) or not np.isfinite([anchor, axis]).all():
            raise ValueError(f"Invalid geometry metadata for {name}")
        if kind != "none":
            if not np.isclose(np.linalg.norm(axis), 1., atol=1e-5):
                raise ValueError(f"Symmetric object {name} requires a documented unit axis")
        elif np.linalg.norm(axis) > 1e-8:
            raise ValueError(f"Asymmetric object {name} must have zero symmetry axis")
        anchors.append(anchor); axes.append(axis); kinds.append(kind)
    return np.asarray(anchors), np.asarray(axes), kinds


def angular_errors(achieved: np.ndarray, reference: np.ndarray, axes: np.ndarray,
                   kinds: list[str]) -> np.ndarray:
    """Official framewise geodesic / half-turn minimum / axial tilt definition."""
    dots = np.clip(abs(np.sum(achieved * reference, axis=-1)), 0., 1.)
    errors = 2 * np.arctan2(np.sqrt(np.maximum(0., 1. - dots * dots)), dots)
    ra, rr = rotation_matrices(achieved), rotation_matrices(reference)
    for body, kind in enumerate(kinds):
        if kind == "half_turn":
            symmetry = Rotation.from_rotvec(np.pi * axes[body]).as_matrix()
            relative = ra[:, :, body].swapaxes(-1, -2) @ rr[:, :, body] @ symmetry
            alternative = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(errors.shape[:2])
            errors[:, :, body] = np.minimum(errors[:, :, body], alternative)
        elif kind == "continuous_axial":
            aa, ar = ra[:, :, body] @ axes[body], rr[:, :, body] @ axes[body]
            errors[:, :, body] = np.arctan2(np.linalg.norm(np.cross(aa, ar), axis=-1),
                                           np.clip(np.sum(aa * ar, axis=-1), -1., 1.))
    return errors


def align_initial_scene(achieved: dict[str, Any], reference: dict[str, Any], axes: np.ndarray,
                        kinds: list[str]) -> tuple[dict[str, Any], list[list]]:
    """Exactly one initial-scene rigid transform per world, following official Track 3."""
    result = dict(achieved)
    result["object_positions"] = achieved["object_positions"].copy()
    result["object_quaternions_wxyz"] = achieved["object_quaternions_wxyz"].copy()
    a, r = achieved["object_positions"], reference["object_positions"]
    ra, rr = rotation_matrices(achieved["object_quaternions_wxyz"]), rotation_matrices(reference["object_quaternions_wxyz"])
    fixed = [i for i, kind in enumerate(kinds) if kind == "none"]
    transforms = []
    def basis(axis, direction):
        perpendicular = direction - axis * np.dot(axis, direction)
        if np.linalg.norm(perpendicular) < 1e-8:
            perpendicular = np.eye(3)[np.argmin(abs(axis))].copy()
            perpendicular -= axis * np.dot(axis, perpendicular)
        perpendicular /= np.linalg.norm(perpendicular)
        return np.column_stack((perpendicular, np.cross(axis, perpendicular), axis))
    for world in range(a.shape[1]):
        if fixed or a.shape[2] == 1:
            anchor = fixed[0] if fixed else 0
            rotation = rr[0, 0, anchor] @ ra[0, world, anchor].T
        else:
            rotation = basis(rr[0, 0, 0] @ axes[0], r[0, 0, 1] - r[0, 0, 0]) @ basis(
                ra[0, world, 0] @ axes[0], a[0, world, 1] - a[0, world, 0]).T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = r[0, 0, 0] - rotation @ a[0, world, 0]
        subset = dict(achieved, object_positions=a[:, world:world+1],
                      object_quaternions_wxyz=achieved["object_quaternions_wxyz"][:, world:world+1])
        moved = apply_world_transform(subset, transform)
        result["object_positions"][:, world:world+1] = moved["object_positions"]
        result["object_quaternions_wxyz"][:, world:world+1] = moved["object_quaternions_wxyz"]
        transforms.append(transform.tolist())
    return result, transforms


def score_aligned(achieved: dict[str, Any], reference: dict[str, Any], axes: np.ndarray, kinds: list[str]) -> dict[str, Any]:
    positions = np.linalg.norm(achieved["object_positions"] - reference["object_positions"], axis=-1)
    angles = angular_errors(achieved["object_quaternions_wxyz"], reference["object_quaternions_wxyz"], axes, kinds)
    mean_positions, mean_angles = positions.mean(axis=0), angles.mean(axis=0)
    passes = (mean_positions < POSITION_THRESHOLD_M) & (mean_angles < ORIENTATION_THRESHOLD_RAD)
    return dict(mp_sr=float(np.mean(np.all(passes, axis=-1))), per_world_success=np.all(passes, axis=-1).tolist(),
                objects={name: dict(mean_position_error_m=mean_positions[:, body].tolist(),
                                    mean_position_error_cm=(100 * mean_positions[:, body]).tolist(),
                                    mean_orientation_error_degrees=np.rad2deg(mean_angles[:, body]).tolist(),
                                    success_per_world=passes[:, body].tolist(),
                                    max_position_error_m=positions[:, :, body].max(axis=0).tolist(),
                                    max_orientation_error_degrees=np.rad2deg(angles[:, :, body].max(axis=0)).tolist())
                         for body, name in enumerate(reference["object_names"])})


def evaluate_arrays(reference: dict[str, Any], trajectory: dict[str, Any], geometry: dict[str, Any],
                    recording_to_reference: np.ndarray | None = None) -> dict[str, Any]:
    reference, trajectory = validate_poses(reference), validate_poses(trajectory)
    if reference["object_positions"].shape[1] != 1:
        raise ValueError("Reference must contain a single world")
    trajectory = resample_to_reference(trajectory, reference)
    transform = np.eye(4) if recording_to_reference is None else np.asarray(recording_to_reference, float)
    trajectory = apply_world_transform(trajectory, transform)
    anchors, axes, kinds = validate_geometry(reference["object_names"], geometry)
    # Both sides describe the same source mesh body frame, so no body registration is fitted.
    for data in (reference, trajectory):
        data["object_positions"] += np.einsum("twoij,oj->twoi", rotation_matrices(data["object_quaternions_wxyz"]), anchors)
    strict = score_aligned(trajectory, reference, axes, kinds)
    registered, initial_transforms = align_initial_scene(trajectory, reference, axes, kinds)
    official = score_aligned(registered, reference, axes, kinds)
    return dict(metric="MP-SR", reference_proxy_mp_sr=official["mp_sr"], declared_frame_mp_sr=strict["mp_sr"],
                official_style_reference_proxy=official, declared_frame_reference_proxy=strict,
                estimated=True, leaderboard_score=False,
                ground_truth="immutable final_recon reconstructed object poses; confidential ground truth unavailable",
                physics_validation="separate; MP-SR alone does not verify physically valid robot control",
                position_threshold_m=POSITION_THRESHOLD_M, orientation_threshold_degrees=30., strict_thresholds=True,
                averaging="framewise symmetry error, then arithmetic mean over all original frames per object; every object must pass",
                episode_aggregation="mean episode MP-SR; episodes have equal weight regardless of duration",
                frames=len(reference["timestamps_seconds"]), worlds=trajectory["object_positions"].shape[1],
                horizon_seconds=[float(reference["timestamps_seconds"][0]), float(reference["timestamps_seconds"][-1])],
                full_horizon=True, temporal_warp=False, per_object_pose_alignment=False,
                recording_to_reference_transform=transform.tolist(),
                official_initial_scene_transforms=initial_transforms, geometry=geometry,
                metric_source=dict(url=KIT_URL, sha256=KIT_SHA256,
                                   functions="v2dlb/track3_pose.py:_symmetry_orientation_errors,_align_registered_scene,_registered_pose_metric"))


def evaluate_files(reference_dir: str | Path, trajectory_path: str | Path,
                   manifest_path: str | Path | None = None, scene_path: str | Path | None = None,
                   geometry_path: str | Path | None = None) -> dict[str, Any]:
    reference_dir, trajectory_path = Path(reference_dir), Path(trajectory_path)
    geometry = json.loads(Path(geometry_path).read_text()) if geometry_path else geometry_from_reference(reference_dir)
    for name, entry in geometry["objects"].items():
        if "mesh_sha256" in entry:
            mesh_path = Path(entry["mesh_path"])
            if not mesh_path.is_file() or sha256(mesh_path) != entry["mesh_sha256"]:
                raise ValueError(f"Reference mesh changed since metric geometry was frozen: {name}")
    reference_file = reference_dir / "object_poses.npz" if reference_dir.is_dir() else reference_dir
    reference_hash = sha256(reference_file)
    transform = np.eye(4)
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if "source_from_setup" not in manifest:
            raise ValueError("Adapter manifest must explicitly declare source_from_setup")
        if (reference_dir.is_dir() and "source_object_poses_sha256" in manifest
                and reference_hash != manifest["source_object_poses_sha256"]):
            raise ValueError("Reference object poses changed since the adapter manifest was frozen")
        transform = np.asarray(manifest["source_from_setup"], float)
    report = evaluate_arrays(load_reference(reference_dir), load_trajectory(trajectory_path), geometry, transform)
    report["provenance"] = dict(reference=str(reference_file.resolve()), reference_sha256=reference_hash,
                                trajectory=str(trajectory_path.resolve()), trajectory_sha256=sha256(trajectory_path))
    for label, path in (("adapter_manifest", manifest_path), ("scene", scene_path), ("geometry", geometry_path)):
        if path:
            report["provenance"][label] = str(Path(path).resolve())
            report["provenance"][label + "_sha256"] = sha256(Path(path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--manifest", type=Path, help="Adapter manifest declaring source_from_setup")
    parser.add_argument("--scene", type=Path, help="Record exact scene path and digest as provenance")
    parser.add_argument("--geometry", type=Path, help="Reviewed body anchors/axes; defaults to documented mesh estimates")
    parser.add_argument("--prepare-geometry", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.prepare_geometry:
        report = geometry_from_reference(args.reference_dir)
    else:
        if args.trajectory is None:
            parser.error("--trajectory is required for evaluation")
        report = evaluate_files(args.reference_dir, args.trajectory, args.manifest, args.scene, args.geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("reference_proxy_mp_sr", "declared_frame_mp_sr", "frames", "full_horizon") if key in report}))


if __name__ == "__main__":
    main()
