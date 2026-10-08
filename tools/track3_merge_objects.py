#!/usr/bin/env python3
"""Add a separately reconstructed object to a Vega motion_v1 reference.

The two reconstructions use the same input video but need not share a world
origin.  The secondary object-to-camera pose is therefore composed with the
primary reconstruction's camera-to-world pose on every source frame.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


def _rotation_track(matrices: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrices)
    rotation = u @ vh
    negative = np.linalg.det(rotation) < 0
    u[negative, :, -1] *= -1
    return u @ vh


def _write_assets(source: Path, destination: Path, object_name: str, scale: float) -> tuple[Path, Path, float]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{object_name}.obj"
    vertices = []
    with source.open() as reader, target.open("w") as writer:
        for line in reader:
            if line.startswith("v "):
                values = line.split()
                xyz = np.array([float(value) for value in values[1:4]]) * scale
                vertices.append(xyz)
                writer.write("v " + " ".join(f"{value:.8f}" for value in xyz))
                writer.write(" " + " ".join(values[4:]) if len(values) > 4 else "")
                writer.write("\n")
            elif line.startswith("mtllib "):
                writer.write(f"mtllib {object_name}_{line.split(maxsplit=1)[1].strip()}\n")
            else:
                writer.write(line)
    if not vertices:
        raise ValueError(f"Mesh has no vertices: {source}")
    for texture in (*source.parent.glob("*.png"), *source.parent.glob("*.jpg")):
        shutil.copy2(texture, destination / f"{object_name}_{texture.name}")
    for material in source.parent.glob("*.mtl"):
        content = material.read_text()
        for texture in (*source.parent.glob("*.png"), *source.parent.glob("*.jpg")):
            content = content.replace(texture.name, f"{object_name}_{texture.name}")
        (destination / f"{object_name}_{material.name}").write_text(content)
    vertex_array = np.asarray(vertices)
    minimum, maximum = vertex_array.min(axis=0), vertex_array.max(axis=0)
    center = (minimum + maximum) / 2
    extents = np.maximum(maximum - minimum, 1e-4)
    mass = 0.3
    inertia = mass * np.array([
        extents[1] ** 2 + extents[2] ** 2,
        extents[0] ** 2 + extents[2] ** 2,
        extents[0] ** 2 + extents[1] ** 2,
    ]) / 12
    urdf = destination / f"{object_name}.urdf"
    urdf.write_text(
        f'<?xml version="1.0"?><robot name="{object_name}"><link name="object">'
        f'<inertial><origin xyz="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" rpy="0 0 0"/>'
        f'<mass value="{mass}"/><inertia ixx="{inertia[0]:.8f}" ixy="0" ixz="0" '
        f'iyy="{inertia[1]:.8f}" iyz="0" izz="{inertia[2]:.8f}"/></inertial>'
        f'<visual><geometry><mesh filename="{target.name}"/></geometry></visual>'
        f'<collision><geometry><mesh filename="{target.name}"/></geometry></collision>'
        '</link></robot>\n'
    )
    return target, urdf, float(np.linalg.norm(vertex_array, axis=1).max())


def merge(
    primary_motion: Path,
    primary_bundle: Path,
    secondary_bundle: Path,
    report_path: Path,
    output: Path,
    asset_dir: Path,
    secondary_name: str,
    source_fps: float,
    container_asset_dir: str,
) -> None:
    table = pq.read_table(primary_motion)
    if table.num_rows != 1:
        raise ValueError("Expected one motion_v1 row")
    row = table.to_pylist()[0]
    if row["schema_version"] != "motion_v1" or len(row["object_body_names"]) != 1:
        raise ValueError("Expected a single-object motion_v1 reference")
    if secondary_name in row["object_body_names"]:
        raise ValueError(f"Duplicate object name: {secondary_name}")
    with np.load(primary_bundle / "result.npz") as primary, np.load(secondary_bundle / "result.npz") as secondary:
        camera_world = np.asarray(primary["camera_to_world_transform"], dtype=np.float64)
        object_camera = np.asarray(secondary["object_to_camera_transform"], dtype=np.float64)
        if camera_world.shape != object_camera.shape or camera_world.shape[1:] != (4, 4):
            raise ValueError(f"Camera/object frame mismatch: {camera_world.shape}, {object_camera.shape}")
        if np.count_nonzero(secondary["object_is_valid"]) == 0:
            raise ValueError("Secondary object was never tracked")
        object_world = camera_world @ object_camera
        embedded_scale = float(np.cbrt(np.linalg.det(object_world[:, :3, :3])).mean())
        object_scale = float(secondary["object_scale"])
        if not np.isfinite(embedded_scale) or embedded_scale <= 0:
            raise ValueError("Invalid secondary object mesh scale")
    report = json.loads(report_path.read_text())
    place = np.asarray(report["placement"]["T_place"], dtype=np.float64)
    scene_scale = float(report.get("scene_scale", 1.0))
    object_world[:, :3, 3] *= scene_scale
    placed = place @ object_world
    rotations = _rotation_track(placed[:, :3, :3])
    positions = placed[:, :3, 3]
    frames = len(row["object_body_position"])
    target_fps = float(row["fps"])
    source_times = np.arange(len(positions)) / source_fps
    target_times = np.clip(np.arange(frames) / target_fps, 0, source_times[-1])
    position = np.stack([np.interp(target_times, source_times, positions[:, axis]) for axis in range(3)], axis=1)
    quaternion = Slerp(source_times, Rotation.from_matrix(rotations))(target_times).as_quat(scalar_first=True)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Secondary object has nonfinite pose values")
    mesh, urdf, radius = _write_assets(
        secondary_bundle / "mesh.obj", asset_dir, secondary_name,
        scene_scale * embedded_scale * object_scale,
    )
    if not container_asset_dir.startswith("/"):
        raise ValueError("container_asset_dir must be absolute")
    updates = {
        "object_body_names": row["object_body_names"] + [secondary_name],
        "safe_object_body_names": row["safe_object_body_names"] + [secondary_name],
        "object_mesh_paths": row["object_mesh_paths"] + [f"{container_asset_dir}/{mesh.name}"],
        "object_urdf_paths": row["object_urdf_paths"] + [f"{container_asset_dir}/{urdf.name}"],
        "object_mesh_radius": row["object_mesh_radius"] + [radius],
        "object_body_position": [[*first, second.tolist()] for first, second in zip(row["object_body_position"], position, strict=True)],
        "object_body_wxyz": [[*first, second.tolist()] for first, second in zip(row["object_body_wxyz"], quaternion, strict=True)],
    }
    for name, value in updates.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"Missing motion_v1 field: {name}")
        table = table.set_column(index, name, pa.array([value], type=table.schema.field(name).type))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, output)
    print(f"merged {row['object_body_names'][0]} + {secondary_name}: {output} ({frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary_motion", "primary_bundle", "secondary_bundle", "report", "output", "asset_dir", "secondary_name", "container_asset_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    args = parser.parse_args()
    merge(
        Path(args.primary_motion), Path(args.primary_bundle), Path(args.secondary_bundle),
        Path(args.report), Path(args.output), Path(args.asset_dir), args.secondary_name,
        args.source_fps, args.container_asset_dir,
    )


if __name__ == "__main__":
    main()
