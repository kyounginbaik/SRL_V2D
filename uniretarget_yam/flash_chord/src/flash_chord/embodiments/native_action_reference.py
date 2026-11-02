"""Native spring targets for actions, separate from geometric references.

Conversion happens at reference construction, before simulation/capture.
Native finger preload is preserved; observations and rewards keep geometry.
"""
import ast
import numpy as np
from scipy.spatial.transform import Rotation


def native_action_arrays(reference,layout,sides):
    sides=tuple(sides);target=np.asarray(reference.joint_target)
    if (set(sides)!=set(layout.sides) or len(sides)!=len(layout.sides)
            or target.shape!=(reference.num_frames,layout.num_joint_dof) or not np.isfinite(target).all()):
        raise ValueError('finite native targets and the complete named hand order required')
    positions=[];orientations=[];fingers=[];used=[]
    for side in sides:
        hand=layout.hand(side);p=tuple(hand.wrist_pos_dof_ids);r=tuple(hand.wrist_orient_dof_ids);f=tuple(hand.finger_dof_ids)
        ids=(*p,*r,*f)
        if (len(p)!=3 or len(r)!=3 or hand.arm_dof_ids
                or any(not isinstance(i,(int,np.integer)) or i<0 or i>=layout.num_joint_dof for i in ids)
                or len(set(ids))!=len(ids) or set(ids).intersection(used)):
            raise ValueError('native_targets requires disjoint named floating-wrist and finger DOFs')
        used.extend(ids);positions.append(target[:,p])
        orientations.append(Rotation.from_rotvec(np.array(target[:,r],float,copy=True)).as_quat()[:,[3,0,1,2]])
        fingers.append(target[:,f])
    return dict(wrist_pos_w=np.stack(positions,axis=1),wrist_quat_wxyz=np.stack(orientations,axis=1),finger_joint_pos=np.concatenate(fingers,axis=1))


def _patch(source,marker,replacements):
    if marker in source:return source
    if any(source.count(old)!=1 for old,_ in replacements):raise ValueError('unsupported native-action reference insertion point')
    for old,new in replacements:source=source.replace(old,new,1)
    ast.parse(source);return source


def patch_binding_text(source):
    """Apply the opt-in binding to a frozen runtime without dropping its patches."""
    source=_patch(source,'# NATIVE_ACTION_REFERENCE_V1',[
        ('    finger_joint_pos: wp.array\n','    finger_joint_pos: wp.array\n    # NATIVE_ACTION_REFERENCE_V1\n    reference_source: Literal["geometry", "native_targets"] = "geometry"\n'),
        ('        device=None,\n    ) -> "DeviceRobotReference":','        device=None,\n        reference_source: Literal["geometry", "native_targets"] = "geometry",\n    ) -> "DeviceRobotReference":'),
        ('        sides = layout.sides if sides is None else tuple(sides)\n','        if reference_source not in ("geometry", "native_targets"):\n            raise ValueError("reference_source must be geometry or native_targets")\n        sides = layout.sides if sides is None else tuple(sides)\n'),
        ('        finger_joint_pos = np.concatenate([hand.finger_joint_pos for hand in hands], axis=1)\n',
         '        finger_joint_pos = np.concatenate([hand.finger_joint_pos for hand in hands], axis=1)\n        if reference_source == "native_targets":\n            from .native_action_reference import native_action_arrays\n            native = native_action_arrays(reference, layout, sides)\n            wrist_pos_w = native["wrist_pos_w"]\n            wrist_quat_wxyz = native["wrist_quat_wxyz"]\n            finger_joint_pos = native["finger_joint_pos"]\n'),
        ('            finger_counts=finger_counts,\n','            finger_counts=finger_counts,\n            reference_source=reference_source,\n')])
    return _patch(source,'# NATIVE_REFERENCE_BINDING_CONTRACT_V1',[
        ('    hands: tuple[BoundHandReference, ...]\n','    hands: tuple[BoundHandReference, ...]\n    # NATIVE_REFERENCE_BINDING_CONTRACT_V1\n    required_action_reference_source: Literal["native_targets"] | None = None\n'),
        ('        if self.num_frames <= 0:\n','        if self.required_action_reference_source not in (None, "native_targets"):\n            raise ValueError("unsupported required action reference source")\n        if self.num_frames <= 0:\n')])


def patch_hand_action_text(source):
    source=_patch(source,'# NATIVE_ACTION_REFERENCE_CONFIG_V1',[
        ('    normalized_mapping: Literal["linear", "rational"] = "linear"\n','    normalized_mapping: Literal["linear", "rational"] = "linear"\n    # NATIVE_ACTION_REFERENCE_CONFIG_V1\n    reference_source: Literal["geometry", "native_targets"] = "geometry"\n'),
        ('            sides=layout.sides,\n            device=device,\n        )\n\n        scale',
         '            sides=layout.sides,\n            device=device,\n            reference_source=config.reference_source,\n        )\n\n        scale')])
    anchor='        layout = ActionLayout.from_embodiment(embodiment_layout, sides=sides, num_joint_dof=num_joint_dof)\n'
    return _patch(source,'# NATIVE_REFERENCE_ACTION_CONTRACT_V1',[(anchor,
        '        # NATIVE_REFERENCE_ACTION_CONTRACT_V1\n        required = getattr(reference, "required_action_reference_source", None)\n        if required is not None and config.reference_source != required:\n            raise ValueError(f"robot reference requires action reference_source={required}")\n'+anchor)])


def patch_rollout_text(source):
    return _patch(source,'# NATIVE_ACTION_REFERENCE_PACKED_V1',[
        ('        ref=DeviceRobotReference.build(packed,env.scene.layout,sides=old.sides,device=device)',
         '        # NATIVE_ACTION_REFERENCE_PACKED_V1\n        ref=DeviceRobotReference.build(packed,env.scene.layout,sides=old.sides,device=device,reference_source=getattr(old,"reference_source","geometry"))')])


def patch_snapshot_text(source):
    return _patch(source,'# NATIVE_REFERENCE_SNAPSHOT_CONTRACT_V1',[
        ('    header = dict(format=1, num_frames=binding.num_frames, fps=binding.fps,\n',
         '    # NATIVE_REFERENCE_SNAPSHOT_CONTRACT_V1\n    header = dict(format=1, num_frames=binding.num_frames, fps=binding.fps,\n                  required_action_reference_source=getattr(binding, "required_action_reference_source", None),\n'),
        ('            joint_q=data["joint_q"].copy(), joint_target=data["joint_target"].copy(), hands=hands)',
         '            joint_q=data["joint_q"].copy(), joint_target=data["joint_target"].copy(), hands=hands,\n            required_action_reference_source=header.get("required_action_reference_source"))')])
