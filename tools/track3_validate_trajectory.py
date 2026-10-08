#!/usr/bin/env python3
"""Validate one evaluator rollout against the public Track 3 row roster."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    args = parser.parse_args()
    kit = args.kit.resolve()
    sys.path.insert(0, str(kit))
    from v2dlb.submission_parquet import provenance, read_trajectory

    trajectory = read_trajectory(args.trajectory)
    episode = trajectory.episode_index
    objects = json.loads((kit / "data/track_3_evaluation_objects.json").read_text())["episodes"]
    expected_names = objects.get(str(episode))
    if expected_names is None:
        raise SystemExit(f"episode {episode} is absent from the scored evaluation roster")
    prefix = f"track_3/e{episode:06d}/"
    with (kit / "data/track_3_sample_submission.csv").open(newline="") as stream:
        expected = {row["row_id"] for row in csv.DictReader(stream)
                    if row["row_id"].startswith(prefix)}
    if not expected:
        raise SystemExit(f"episode {episode} has no public row IDs")

    actual = [
        f"track_3/e{episode:06d}/w{int(world):03d}/f{int(frame):06d}/o{int(slot)}"
        for world, frame, slot in zip(
            trajectory.world_index,
            trajectory.frame_index,
            trajectory.object_slot,
            strict=True,
        )
    ]
    if len(actual) != len(set(actual)):
        raise SystemExit("rollout has duplicate world/frame/object rows")
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise SystemExit(f"row roster mismatch: {len(missing)} missing, {len(extra)} extra; "
                         f"examples: missing={missing[:3]}, extra={extra[:3]}")
    if not trajectory.world_valid.all():
        raise SystemExit("rollout has non-finite evaluation worlds")
    if not np.isfinite(trajectory.pose_xyzw).all():
        raise SystemExit("rollout has non-finite poses")
    quaternion_norm = np.linalg.norm(trajectory.pose_xyzw[:, 3:], axis=1)
    if np.max(np.abs(quaternion_norm - 1.0)) > 1e-3:
        raise SystemExit("rollout quaternions are not unit length")
    if trajectory.object_name is None:
        raise SystemExit("rollout does not record object body names")
    for slot, name in enumerate(expected_names):
        found = set(trajectory.object_name[trajectory.object_slot == slot])
        if found != {name}:
            raise SystemExit(f"object slot {slot}: expected {name!r}, found {found}")
    hashes = provenance(trajectory)
    for label in ("checkpoint_sha256", "eval_script_sha256", "eval_code_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", hashes.get(label, "")):
            raise SystemExit(f"missing or invalid {label}")
    print(f"PASS episode_{episode:06d}: {len(actual)} rows, {len(expected_names)} object(s), "
          f"{len(set(trajectory.frame_index))} frames, provenance present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
