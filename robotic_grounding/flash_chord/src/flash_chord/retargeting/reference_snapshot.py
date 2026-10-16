"""Versioned robot-only reference snapshots; object/contact commands are never edited."""

from dataclasses import fields
from pathlib import Path
import hashlib
import io
import json
import os

import numpy as np

from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding


def command_digest(reference):
    """Fingerprint immutable task commands, independently of robot versions."""
    digest = hashlib.sha256()
    arrays = [reference.object_body_pos_w(), reference.object_body_quat_w()]
    for side in reference.sides:
        arrays.extend([reference.contact_pos_w(side), reference.contact_normal_w(side),
                       reference.contact_part_ids(side), reference.contact_active(side)])
    for array in arrays:
        array = np.asarray(array)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_snapshot(path, binding, layout, *, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binding.validate(layout)
    header = dict(format=1, num_frames=binding.num_frames, fps=binding.fps,
                  num_joint_q=binding.num_joint_q, num_joint_dof=binding.num_joint_dof,
                  sides=[h.side for h in binding.hands],
                  joint_names=list(layout.scalar_joints.names), provenance=provenance)
    arrays = dict(header=np.asarray(json.dumps(header, sort_keys=True)),
                  joint_q=binding.joint_q, joint_target=binding.joint_target)
    for hand in binding.hands:
        for field in fields(BoundHandReference):
            if field.name != "side":
                arrays[f"{hand.side}/{field.name}"] = getattr(hand, field.name)
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_archive_command_digest(path, header, snapshot_sha256):
    """Recover omitted provenance only from the original checksum-bound archive record.

    Early DemoArchive snapshots omitted the command digest in their header but
    recorded it in pair.json alongside the snapshot checksum. Absolute paths
    may have moved with the archive; checksum and archive identity remain exact.
    """
    pair_path = path.with_name("pair.json")
    if not pair_path.is_file():
        raise ValueError("snapshot object/contact source commands do not match: missing provenance")
    pair = json.loads(pair_path.read_text())
    provenance = header["provenance"]
    identity = {"step": "environment_steps", "version": "reference_version",
                "origin": "origin", "initial_state_sha256": "initial_state_sha256"}
    if (not isinstance(pair, dict)
            or pair.get("reference_sha256") != snapshot_sha256
            or not isinstance(pair.get("reference"), str)
            or Path(pair["reference"]).name != path.name
            or any(provenance.get(key) is None or provenance[key] != pair.get(value)
                   for key, value in identity.items())):
        raise ValueError("legacy archive pair does not match the reference snapshot checksum/identity")
    source_hash = pair.get("object_command_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("legacy archive pair is missing its object/contact command digest")
    return source_hash


def load_snapshot(path, layout, baseline, *, sha256=None, source_reference=None):
    path = Path(path)
    contents = path.read_bytes()
    snapshot_sha256 = hashlib.sha256(contents).hexdigest()
    if sha256 is not None and snapshot_sha256 != sha256:
        raise ValueError("robot reference snapshot checksum mismatch")
    with np.load(io.BytesIO(contents), allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        source_hash = header["provenance"].get("object_contact_sha256")
        if source_reference is not None:
            if source_hash is None:
                source_hash = _legacy_archive_command_digest(path, header, snapshot_sha256)
            if source_hash != command_digest(source_reference):
                raise ValueError("snapshot object/contact source commands do not match")
        if header["format"] != 1 or header["joint_names"] != list(layout.scalar_joints.names):
            raise ValueError("snapshot robot joint names/order do not match this embodiment")
        if header["num_frames"] != baseline.num_frames or abs(header["fps"] - baseline.fps) > 1e-6:
            raise ValueError("snapshot does not match reference frame window/FPS")
        hands = tuple(BoundHandReference(side=side, **{
            f.name: data[f"{side}/{f.name}"].copy()
            for f in fields(BoundHandReference) if f.name != "side"
        }) for side in header["sides"])
        result = RobotReferenceBinding(**{k: header[k] for k in
            ("num_frames", "fps", "num_joint_q", "num_joint_dof")},
            joint_q=data["joint_q"].copy(), joint_target=data["joint_target"].copy(), hands=hands)
    result.validate(layout)
    return result
