"""Object-relative hand targets and invertible residual control, without physics edits.

The caller supplies the live object pose, an explicit grasp, and the action
mapping. Targets still pass through the simulator's ordinary action/control
path. Moving a target is not a certificate of a physically feasible grasp.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _pose(value):
    pose = np.asarray(value, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError('a finite position + nonzero xyzw quaternion is required')
    return pose[:3], Rotation.from_quat(pose[3:])


@dataclass(frozen=True)
class ProcessedHandTarget:
    wrist_pose_xyzw: np.ndarray
    fingers: np.ndarray


def decode_processed_hand_target(processed, *, finger_count: int) -> ProcessedHandTarget:
    """Decode a HandPoseAction target: XYZ, WXYZ, then finger spring targets.

    Simulator body/joint poses use XYZW; the processed action buffer uses WXYZ.
    Mixing them can invent a large rotational preload in an otherwise static
    grasp. Pass one explicitly selected hand slice, or a batch of such slices.
    Preserve floating dtype and finger overdrive exactly; these are actuator
    commands, not physical joint poses to clip to joint limits.
    """
    if (isinstance(finger_count, (bool, np.bool_))
            or not isinstance(finger_count, (int, np.integer)) or finger_count < 1):
        raise ValueError('a positive integer finger_count is required')
    values = np.asarray(processed)
    if (values.ndim < 1 or values.shape[-1] != 7 + finger_count
            or values.dtype.kind != 'f' or not np.isfinite(values).all()):
        raise ValueError('a finite floating hand target slice with 7 + finger_count values is required')
    if not np.allclose(np.linalg.norm(values[..., 3:7], axis=-1), 1., rtol=0., atol=1e-5):
        raise ValueError('processed hand target must have a unit WXYZ wrist quaternion')
    wrist = np.concatenate((values[..., :3], values[..., [4, 5, 6, 3]]), axis=-1)
    return ProcessedHandTarget(wrist, values[..., 7:].copy())


@dataclass(frozen=True)
class ObjectRelativeGrasp:
    wrist_position_o: tuple[float, float, float]
    wrist_quaternion_o: tuple[float, float, float, float]
    fingers: tuple[float, ...]

    def __post_init__(self):
        _pose(np.r_[self.wrist_position_o, self.wrist_quaternion_o])
        if np.asarray(self.fingers).ndim != 1 or not len(self.fingers) or not np.isfinite(self.fingers).all():
            raise ValueError('finite scalar finger targets are required')


def capture_object_relative_grasp(object_pose, wrist_pose, fingers):
    """Capture matched pose geometry and explicit finger control targets.

    For a loaded physical grip, pass its processed finger targets to preserve
    squeeze preload. Replacing those targets with achieved joint positions can
    remove the spring force holding the object. Save achieved joints separately.
    ``wrist_pose`` normally supplies achieved geometry. To rebase a loaded
    CONTROL target instead, explicitly supply its decoded processed XYZW wrist
    pose as well: using achieved wrist pose drops wrist spring preload too.
    Such a control anchor is not collision geometry, and its retained bias must
    not receive the same load/feedforward compensation a second time.
    """
    op, rotation = _pose(object_pose)
    wp, wrist_rotation = _pose(wrist_pose)
    return ObjectRelativeGrasp(tuple(rotation.inv().apply(wp - op)),
        tuple((rotation.inv() * wrist_rotation).as_quat()), tuple(np.asarray(fingers, float)))


def rebase_grasp_motion(anchor, reference_anchor_object_pose, reference_anchor_wrist_pose,
                        reference_anchor_fingers, reference_object_pose, reference_wrist_pose,
                        reference_fingers):
    """Retain intended in-hand motion after changing the acquisition grasp.

    Let G0 and Gt be the reference hand poses in object coordinates. The new
    relative pose is anchor * inverse(G0) * Gt. Thus a new physical grasp can
    persist through a long contact interval while reference in-hand rotations,
    translations and finger changes continue. The object trajectory is never
    edited. Call conditioned_hand_target with the returned grasp and live pose.
    """
    base = capture_object_relative_grasp(reference_anchor_object_pose,
        reference_anchor_wrist_pose, reference_anchor_fingers)
    current = capture_object_relative_grasp(reference_object_pose, reference_wrist_pose,
        reference_fingers)
    a = Rotation.from_quat(anchor.wrist_quaternion_o)
    b = Rotation.from_quat(base.wrist_quaternion_o)
    c = Rotation.from_quat(current.wrist_quaternion_o)
    offset = b.inv().apply(np.asarray(current.wrist_position_o) - base.wrist_position_o)
    position = np.asarray(anchor.wrist_position_o) + a.apply(offset)
    fingers = np.asarray(anchor.fingers) + np.asarray(current.fingers) - base.fingers
    if np.shape(anchor.fingers) != np.shape(base.fingers) or np.shape(current.fingers) != np.shape(base.fingers):
        raise ValueError('anchor and reference grasp fingers must have matching shapes')
    return ObjectRelativeGrasp(tuple(position), tuple((a * b.inv() * c).as_quat()), tuple(fingers))


def task_trajectory_in_execution_frame(reference_poses, actual_initial_object0, reference_initial_object0):
    """Invert official object-0 first-pose alignment for HAND target generation.

    Return a task-equivalent object path in the execution frame. The observed
    initial pose is fixed before optimization; this is not a per-frame fit to
    failed rollouts. Apply the SAME initial object-0 transform to every object.
    The immutable task trajectory and official scoring targets are never edited.
    """
    poses = np.asarray(reference_poses, float)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not len(poses)
            or not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-8)):
        raise ValueError('finite [frame,7] reference object trajectory required')
    ap, ar = _pose(actual_initial_object0)
    rp, rr = _pose(reference_initial_object0)
    transform = ar * rr.inv()
    return np.c_[ap + transform.apply(poses[:, :3] - rp),
                 (transform * Rotation.from_quat(poses[:, 3:].copy())).as_quat()]


def conditioned_hand_target(grasp, object_pose, *, linear_velocity=None, angular_velocity=None,
                            lead_s=0., max_position_lead_m=.02, max_rotation_lead_rad=.2):
    """Follow the measured object, with bounded optional constant-velocity lead."""
    position, rotation = _pose(object_pose)
    settings = np.asarray([lead_s, max_position_lead_m, max_rotation_lead_rad], float)
    if not np.isfinite(settings).all() or (settings < 0).any():
        raise ValueError('lead and prediction bounds must be finite and nonnegative')
    def prediction(velocity, bound):
        value = np.zeros(3) if velocity is None else np.asarray(velocity, float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('finite three-dimensional world velocities are required')
        value = lead_s * value
        return value * min(1., bound / max(np.linalg.norm(value), 1e-12))
    position = position + prediction(linear_velocity, max_position_lead_m)
    rotation = Rotation.from_rotvec(prediction(angular_velocity, max_rotation_lead_rad)) * rotation
    wrist = np.r_[position + rotation.apply(grasp.wrist_position_o),
                  (rotation * Rotation.from_quat(grasp.wrist_quaternion_o)).as_quat()]
    return wrist, np.asarray(grasp.fingers, float)


def handover_hand_target(grasp, measured_object_pose, reference_object_pose, *, carry_weight, **prediction):
    """Catch at the live pose, then move the grasp along the fixed task path.

    carry_weight=0 follows the measured object during acquisition; weight=1
    follows the immutable reference after the receiver supports it. The caller
    supplies a smooth, contact-triggered transition and enforces target speeds.
    This constructs a hand target and never writes an object command or state.
    """
    actual_position, actual_rotation = _pose(measured_object_pose)
    reference_position, reference_rotation = _pose(reference_object_pose)
    if not np.isfinite(carry_weight) or not 0 <= carry_weight <= 1:
        raise ValueError('carry weight must be finite and in [0,1]')
    position = actual_position + carry_weight * (reference_position - actual_position)
    delta = (reference_rotation * actual_rotation.inv()).as_rotvec()
    rotation = Rotation.from_rotvec(carry_weight * delta) * actual_rotation
    return conditioned_hand_target(grasp, np.r_[position, rotation.as_quat()], **prediction)


def correct_planned_hand_target(planned_wrist_pose, nominal_object_before_command,
                                measured_object_before_command):
    """Condition command i on matched causal object states at boundary i.

    H_target[i] = A_before[i] * inverse(D_before[i]) * H_plan[i].
    Recorded post-transition row i is boundary i+1, so it must not be used as
    D_before[i]. The planned hand command may already anticipate task motion;
    this correction preserves it exactly when live and nominal states agree.
    """
    hp, hr = _pose(planned_wrist_pose)
    dp, dr = _pose(nominal_object_before_command)
    ap, ar = _pose(measured_object_before_command)
    if np.array_equal(nominal_object_before_command, measured_object_before_command):
        return np.asarray(planned_wrist_pose, float).copy()
    rotation = ar * dr.inv()
    return np.r_[ap + rotation.apply(hp - dp), (rotation * hr).as_quat()]


def hand_target_residual(reference_wrist_pose, reference_fingers, target_wrist_pose, target_fingers):
    """World XYZ translation, reference-local XYZ Euler rotation, then fingers.

    The orientation convention matches ref_quat * quat_rpy in residual hand
    actions. A global rotation vector would be incorrect for this interface.
    """
    rp, rr = _pose(reference_wrist_pose)
    tp, tr = _pose(target_wrist_pose)
    rf, tf = np.asarray(reference_fingers, float), np.asarray(target_fingers, float)
    if rf.ndim != 1 or rf.shape != tf.shape or not np.isfinite([rf, tf]).all():
        raise ValueError('matching finite finger vectors are required')
    return np.r_[tp - rp, (rr.inv() * tr).as_euler('xyz'), tf - rf]


@dataclass(frozen=True)
class EncodedResidual:
    action: np.ndarray
    filtered: np.ndarray
    saturated: np.ndarray


def encode_filtered_residual(desired, previous, scale, clip, *, ema, mapping, input_scale=None):
    """Invert clipping, EMA and raw/linear/rational policy-input processing.

    Return the closest currently reachable physical residual, not an assertion
    that a bounded action can realize an arbitrary target immediately. Inputs
    may include matching leading batch dimensions; parameters broadcast.
    """
    desired, previous, scale, clip = np.broadcast_arrays(*[np.asarray(x, float)
                                                         for x in (desired, previous, scale, clip)])
    if (not all(np.isfinite(x).all() for x in (desired, previous, scale, clip))
            or np.any(scale <= 0) or np.any(clip <= 0) or not np.isfinite(ema) or not 0 <= ema < 1):
        raise ValueError('finite residuals, positive scales/clips, and 0 <= ema < 1 are required')
    if mapping not in ('identity', 'linear', 'rational'):
        raise ValueError('mapping must be identity, linear, or rational')
    target = np.clip(desired, -clip, clip)
    mapped = (target - ema * previous) / ((1 - ema) * scale)
    if mapping == 'identity':
        action = mapped
    else:
        endpoint = np.broadcast_to(np.asarray(input_scale, float), desired.shape)
        if not np.isfinite(endpoint).all() or np.any(endpoint <= 0):
            raise ValueError('positive finite normalized-input endpoint scales are required')
        mapped = np.clip(mapped, -endpoint, endpoint)
        if mapping == 'linear':
            action = mapped / endpoint
        else:
            action = mapped / (1 + np.abs(mapped) * (1 - 1 / endpoint))
        action = np.clip(action, -1., 1.)
    filtered = np.clip(ema * previous + (1 - ema) * scale * mapped, -clip, clip)
    return EncodedResidual(action, filtered, np.abs(filtered - desired) > 1e-7)
