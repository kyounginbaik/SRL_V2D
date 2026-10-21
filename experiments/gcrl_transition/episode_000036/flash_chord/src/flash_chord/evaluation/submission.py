"""Export a policy rollout in the Track 3 submission kit's trajectory schema."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np


def write_object_trajectories_parquet(
    path: str | Path,
    rollout: object,
    *,
    checkpoint_sha256: str,
    eval_script_sha256: str,
    eval_code_sha256: str,
    reference_parquet: str | Path,
) -> Path:
    """Write step/world/body-ordered achieved and reference poses.

    The submission packer, rather than this writer, decides whether the frame
    roster and object order match a particular challenge episode.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    achieved = np.asarray(rollout.achieved_pose_w, dtype=np.float32)
    reference = np.asarray(rollout.reference_pose_w, dtype=np.float32)
    if achieved.ndim != 4 or achieved.shape[-1] != 7:
        raise ValueError(f"achieved poses must be [step, world, body, 7], got {achieved.shape}")
    steps, worlds, bodies, _ = achieved.shape
    if reference.shape != (steps, bodies, 7):
        raise ValueError(f"reference poses must be [{steps}, {bodies}, 7], got {reference.shape}")
    names = tuple(rollout.object_body_names)
    owners = np.asarray(rollout.body_object_ids, dtype=np.int32)
    if len(names) != bodies or owners.shape != (bodies,):
        raise ValueError("object names and owner IDs must have one entry per body")
    if not np.isfinite(achieved).all() or not np.isfinite(reference).all():
        raise ValueError("trajectory poses must be finite")
    if len(set(names)) != bodies:
        raise ValueError("object body names must be unique")

    count = steps * worlds * bodies
    achieved_flat = achieved.reshape(count, 7)
    reference_flat = np.broadcast_to(reference[:, None], achieved.shape).reshape(count, 7)
    invalid_worlds = np.asarray(rollout.non_finite_worlds, dtype=np.int64)
    if np.any((invalid_worlds < 0) | (invalid_worlds >= worlds)):
        raise ValueError("non-finite world ID is outside the rollout")
    world_valid = np.ones(worlds, dtype=np.bool_)
    world_valid[invalid_worlds] = False

    columns: dict[str, object] = {
        "environment_id": np.broadcast_to(np.arange(worlds, dtype=np.int32)[None, :, None],
                                           (steps, worlds, bodies)).reshape(count),
        "step": np.broadcast_to(np.arange(steps, dtype=np.int32)[:, None, None],
                                (steps, worlds, bodies)).reshape(count),
        "reference_frame": np.broadcast_to(
            (np.arange(steps, dtype=np.int32) + int(rollout.start_frame))[:, None, None],
            (steps, worlds, bodies),
        ).reshape(count),
        "time_s": np.broadcast_to(
            (np.arange(steps, dtype=np.float64) / float(rollout.control_fps))[:, None, None],
            (steps, worlds, bodies),
        ).reshape(count),
        "object_id": np.tile(owners, steps * worlds),
        "object_body_id": np.tile(np.arange(bodies, dtype=np.int32), steps * worlds),
        "object_body_name": np.tile(np.asarray(names, dtype=object), steps * worlds).tolist(),
        "environment_valid": np.broadcast_to(world_valid[None, :, None],
                                               (steps, worlds, bodies)).reshape(count),
    }
    suffixes = ("position_x", "position_y", "position_z", "quaternion_x",
                "quaternion_y", "quaternion_z", "quaternion_w")
    for prefix, poses in (("achieved", achieved_flat), ("reference", reference_flat)):
        for index, suffix in enumerate(suffixes):
            columns[f"{prefix}_{suffix}"] = poses[:, index]

    reference_path = str(reference_parquet)
    episode_match = re.search(r"episode_(\d+)", reference_path)
    metadata = {
        "flash_chord.schema": "flash_chord_object_trajectories_v1",
        "flash_chord.world_count": str(worlds),
        "flash_chord.quaternion_order": "xyzw",
        "flash_chord.position_unit": "meter",
        "flash_chord.reference_parquet": reference_path,
        "flash_chord.checkpoint_sha256": checkpoint_sha256,
        "flash_chord.eval_script_sha256": eval_script_sha256,
        "flash_chord.eval_code_sha256": eval_code_sha256,
    }
    if episode_match:
        metadata["flash_chord.episode_index"] = str(int(episode_match.group(1)))
    table = pa.table(columns).replace_schema_metadata(metadata)

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
