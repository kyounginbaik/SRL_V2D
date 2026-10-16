"""Validate optional, episode-specific initial-state evidence before freezing it."""
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_initial_state_report(path, *, episode, source_sha256, frozen_directory, object_names,
                                  object_urdf_paths):
    """Require exact inputs and CPU evidence; GPU certification remains separate.

    This is optional per-episode evidence, not a passive-motion requirement for
    every manipulation task. It never modifies trajectories or collision assets.
    """
    path, frozen = Path(path), Path(frozen_directory)
    report = json.loads(path.read_text())
    if report.get('schema_version') != 1 or report.get('episode') != episode:
        raise ValueError('Initial-state report schema or episode mismatch')
    if report.get('source_reference_sha256') != source_sha256:
        raise ValueError('Initial-state report belongs to a different reference')
    if report.get('source_and_frozen_object_trajectories_identical') is not True:
        raise ValueError('Initial-state report does not preserve the reference targets')
    for filename, key in (('support.usda', 'support_sha256'), ('support_report.json', 'support_report_sha256')):
        if _digest(frozen/'assets'/filename) != report.get(key):
            raise ValueError(f'Initial-state support evidence changed: {filename}')
    closure = report.get('object_asset_closure_sha256', {})
    poses = report.get('object_initial_pose_overrides', {})
    if not isinstance(poses, dict) or not isinstance(closure, dict) or set(poses) != set(object_names) or set(closure) != set(object_names):
        raise ValueError('Initial-state poses/assets must name exactly every object')
    if len(object_urdf_paths) != len(object_names):
        raise ValueError('Initial-state object URDF roster differs from object names')
    for name, urdf_path in zip(object_names, object_urdf_paths, strict=True):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', name):
            raise ValueError(f'Object name is not a portable Hydra map key: {name}')
        pose = poses[name]
        if not isinstance(pose, dict) or set(pose) != {'position', 'quaternion_wxyz'}:
            raise ValueError(f'Initial-state pose schema mismatch: {name}')
        for key, width in (('position', 3), ('quaternion_wxyz', 4)):
            value = pose[key]
            if (not isinstance(value, list) or len(value) != width
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
                raise ValueError(f'Invalid initial-state {key}: {name}')
        if abs(math.sqrt(sum(x*x for x in pose['quaternion_wxyz']))-1.) > 1e-4:
            raise ValueError(f'Initial-state quaternion must be unit length: {name}')
        if not closure[name]:
            raise ValueError(f'Missing initial-state asset closure: {name}')
        urdf = Path(urdf_path)
        if not urdf.is_absolute():
            urdf = frozen/urdf
        urdf = urdf.resolve()
        required = {urdf}
        for mesh in ET.parse(urdf).findall('.//mesh'):
            filename = mesh.attrib.get('filename')
            if not filename or '://' in filename:
                raise ValueError(f'Unresolved initial-state URDF mesh: {urdf}')
            required.add((urdf.parent/filename).resolve())
        supplied = {(frozen/relative).resolve() for relative in closure[name]}
        if not required.issubset(supplied):
            raise ValueError(f'Initial-state asset closure omits URDF dependencies: {name}')
        for relative, expected in closure[name].items():
            candidate = (frozen/relative).resolve()
            if not candidate.is_relative_to((frozen/'assets').resolve()) or _digest(candidate) != expected:
                raise ValueError(f'Initial-state object asset changed: {relative}')
    passive = report.get('passive_validation', {})
    gate = passive.get('exact_command_index_reference_gate', {})
    if (report.get('cpu_physics_passed') is not True
            or report.get('detachment_validation', {}).get('passed') is not True
            or gate.get('target_unchanged') is not True
            or gate.get('pass_every_sample_first2s') is not True
            or passive.get('duration_seconds', 0) < 2.
            or passive.get('physics', {}).get('assistance') != 0):
        raise ValueError('Initial-state CPU stability, target tracking, or detachment evidence failed')
    limits = gate.get('limits', {})
    if limits != {'position_m': .05, 'orientation_rad': math.radians(20), 'persistence_frames': 3}:
        raise ValueError('Initial-state tracking evidence uses different failure thresholds')
    for key, limit in (('max_position_error_first2s_m', .05),
                       ('max_orientation_error_first2s_rad', math.radians(20))):
        values = gate.get(key, [])
        if len(values) != len(object_names) or any(not math.isfinite(x) or x < 0 or x > limit for x in values):
            raise ValueError(f'Initial-state tracking gate contradicted by {key}')
    return dict(object_initial_pose_overrides=poses, initial_state_report_sha256=_digest(path),
                initial_state_scope='frame_zero_only', initial_state_cpu_physics_passed=True,
                initial_state_gpu_physics_passed=report.get('gpu_physics_passed') is True,
                reference_commands_modified=False)
