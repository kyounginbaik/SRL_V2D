"""Validate real Track 3 rollouts against the participant-visible row roster."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROSTER = HERE / "public_roster.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def episode_index(value):
    text = str(value)
    match = re.fullmatch(r"(?:episode_)?(\d+)", text)
    if not match:
        raise ValueError(f"Invalid episode identifier: {value!r}")
    return int(match.group(1))


def load_roster(path=ROSTER):
    return json.loads(Path(path).read_text())["episodes"]


def validate_kit_roster(kit, roster=None):
    """Do not silently package against a different official challenge release."""
    roster = load_roster() if roster is None else roster
    kit = Path(kit)
    names = json.loads((kit / "data/track_3_evaluation_objects.json").read_text())["episodes"]
    if names != {key: entry["object_names"] for key, entry in roster.items()}:
        raise ValueError("Official kit object roster differs from the frozen public roster")
    wanted = {
        f"track_3/e{int(key):06d}/w{world:03d}/f{frame:06d}/o{slot}"
        for key, entry in roster.items() for world in entry["world_indices"]
        for frame in entry["frame_indices"] for slot in range(len(entry["object_names"]))
    }
    with (kit / "data/track_3_sample_submission.csv").open(newline="") as stream:
        actual = [row["row_id"] for row in csv.DictReader(stream)]
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError("Official kit frame/world/object row roster differs from the frozen roster")


def validate_trajectory(path, *, episode=None, checkpoint_sha256=None, roster=None):
    """Reject partial, assisted-by-contract, malformed or misattributed exports.

    The parquet proves pose/schema/provenance consistency. The export helper's
    fixed evaluation composition, not a parquet field, establishes VOC=0.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    roster = load_roster() if roster is None else roster
    table = pq.read_table(path)
    metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
    expected_metadata = {"flash_chord.schema": "flash_chord_object_trajectories_v1",
                         "flash_chord.world_count": "1",
                         "flash_chord.quaternion_order": "xyzw",
                         "flash_chord.position_unit": "meter"}
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path.name}: expected {key}={value!r}")
    recorded_episode = episode_index(metadata.get("flash_chord.episode_index", ""))
    if episode is not None and recorded_episode != episode_index(episode):
        raise ValueError("Export belongs to a different episode")
    entry = roster.get(str(recorded_episode))
    if entry is None:
        raise ValueError(f"Episode {recorded_episode} is outside the scored public roster")
    keys = ("environment_id", "reference_frame", "object_body_id")
    for key in (*keys, "object_body_name", "environment_valid"):
        if key not in table.column_names or table[key].null_count:
            raise ValueError(f"Missing or null trajectory column {key}")
    index = [np.asarray(table[key]) for key in keys]
    if any(not np.issubdtype(values.dtype, np.integer) for values in index):
        raise ValueError("World/frame/object identifiers must be integers")
    actual = list(zip(*(values.tolist() for values in index), strict=True))
    expected = {(world, frame, slot) for world in entry["world_indices"]
                for frame in entry["frame_indices"] for slot in range(len(entry["object_names"]))}
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate world/frame/object rows")
    if set(actual) != expected:
        raise ValueError(f"Row roster mismatch: {len(expected-set(actual))} missing, "
                         f"{len(set(actual)-expected)} extra")
    valid = np.asarray(table["environment_valid"])
    if valid.dtype != np.bool_ or not valid.all():
        raise ValueError("Trajectory contains invalid simulation worlds")
    names = np.asarray(table["object_body_name"])
    for slot, name in enumerate(entry["object_names"]):
        if set(names[index[2] == slot]) != {name}:
            raise ValueError(f"Object slot {slot} must contain {name!r}")
    suffixes = ("position_x", "position_y", "position_z", "quaternion_x",
                "quaternion_y", "quaternion_z", "quaternion_w")
    for prefix in ("achieved", "reference"):
        columns = [f"{prefix}_{suffix}" for suffix in suffixes]
        if any(key not in table.column_names or table[key].null_count for key in columns):
            raise ValueError(f"Missing or null {prefix} pose columns")
        poses = np.stack([np.asarray(table[key]) for key in columns], axis=-1)
        if not np.isfinite(poses).all():
            raise ValueError(f"Non-finite {prefix} poses")
        if np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.0)) > 1e-3:
            raise ValueError(f"Non-unit {prefix} quaternions")
    hashes = {}
    for key in ("checkpoint_sha256", "eval_script_sha256", "eval_code_sha256"):
        value = metadata.get(f"flash_chord.{key}", "")
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Missing or invalid {key}")
        hashes[key] = value
    if checkpoint_sha256 is not None and hashes["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Trajectory checkpoint hash does not match the evaluated checkpoint")
    return dict(valid=True, episode_index=recorded_episode, rows=table.num_rows,
                frames=len(entry["frame_indices"]), worlds=len(entry["world_indices"]),
                objects=entry["object_names"], trajectory_sha256=digest(path), **hashes)


def evaluate_command(helper, checkpoint, episode, output, *, executable, kit=None,
                     selection_criterion="explicit_checkpoint"):
    command = [str(executable), str(Path(helper) / "export_episode.py"),
               "--checkpoint", str(checkpoint), "--episode", str(episode), "--output", str(output),
               "--selection-criterion", selection_criterion]
    if kit:
        command += ["--kit", str(kit)]
    return command
