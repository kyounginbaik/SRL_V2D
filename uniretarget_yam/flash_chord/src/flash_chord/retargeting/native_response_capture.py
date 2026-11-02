"""Extract named-hand actuator identification data from physical rollouts.

This selects a declared reference group, never the best-performing worlds.
Independent hand load is evidence for model fitting, not task success or proof
that environmental support is absent.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def native_response_capture(data, schema, *, hand, object_id, group):
    palms = np.asarray(data['palm_pose'])
    native = np.asarray(data['commanded_robot_joint_target'])
    counts = np.asarray(data['hand_object_contact_count'])
    loads = np.asarray(data['hand_object_support_mg'])
    groups = np.asarray(data['world_reference_group'])
    palm_sides = list(data['palm_sides'])
    contact_sides = list(data['physical_contact_sides'])
    if (palms.ndim != 4 or palms.shape[-1] != 7 or native.ndim != 3
            or counts.ndim != 4 or loads.shape != counts.shape
            or palms.shape[:2] != native.shape[:2] or palms.shape[:2] != counts.shape[:2]
            or groups.shape != (palms.shape[1],) or not np.issubdtype(groups.dtype, np.integer)
            or len(palm_sides) != palms.shape[2] or len(set(palm_sides)) != len(palm_sides)
            or len(contact_sides) != counts.shape[2] or len(set(contact_sides)) != len(contact_sides)
            or hand not in palm_sides or hand not in contact_sides
            or not isinstance(object_id, (int, np.integer)) or not 0 <= object_id < counts.shape[3]
            or not isinstance(group, (int, np.integer))):
        raise ValueError('aligned named hand, object, native controls and reference groups required')
    execution = np.asarray(data['execution_frame_ids'])
    contact = np.asarray(data['contact_frame_ids'])
    if (execution.shape != (len(palms),) or not np.array_equal(execution, contact)
            or not np.array_equal(execution, np.arange(len(palms)))):
        raise ValueError('contiguous matched execution/contact clocks required')
    ids = np.flatnonzero(groups == group)
    if not len(ids):
        raise ValueError('declared reference group has no worlds')
    pids = list(schema['wrist_pos_dof_ids'])
    rids = list(schema['wrist_orient_dof_ids'])
    if (len(pids) != 3 or len(rids) != 3 or len(set(pids + rids)) != 6
            or any(not isinstance(i, (int, np.integer)) or not 0 <= i < native.shape[2] for i in pids + rids)):
        raise ValueError('six distinct native wrist target coordinates required')
    # Select scalar side/object axes before the world index. Separated advanced
    # indices otherwise silently transpose time and world axes in NumPy.
    actual = palms[:, :, palm_sides.index(hand)][:, ids]
    commands = native[:, ids]
    rv = commands[:, :, rids]
    finite_rotation = np.isfinite(rv).all(-1)
    quaternion = np.full((*rv.shape[:2], 4), np.nan)
    quaternion[finite_rotation] = Rotation.from_rotvec(rv[finite_rotation]).as_quat()
    command_pose = np.concatenate([commands[:, :, pids], quaternion], axis=-1)
    h = contact_sides.index(hand)
    selected_counts = counts[:, ids]
    selected_loads = loads[:, ids]
    own = selected_counts[:, :, h, object_id]
    other_hands = np.delete(selected_counts[..., object_id], h, axis=2)
    other_objects = np.delete(selected_counts[:, :, h, :], object_id, axis=2)
    load = selected_loads[:, :, h, object_id]
    supported = ((own > 0) & (other_hands == 0).all(2)
                 & (other_objects == 0).all(2) & np.isfinite(load) & (load > .5))
    return actual, command_pose, supported, commands, ids
