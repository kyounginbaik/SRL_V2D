"""Rigidly transport floating-hand geometry and native targets together."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .coupled_transport import transport_poses


def transfer_floating_reference(q,target,source_object,control_object,layout):
    q=np.asarray(q);target=np.asarray(target)
    if (q.ndim!=2 or q.shape[1]!=layout.num_joint_q or target.shape!=(len(q),layout.num_joint_dof)
            or not np.isfinite(q).all() or not np.isfinite(target).all()):
        raise ValueError('finite named floating-hand geometry and native targets required')
    geometry=[];commands=[]
    for hand in layout.hands:
        if len(hand.wrist_pos_q_ids)!=3 or len(hand.wrist_orient_dof_ids)!=3 or hand.wrist_orient_q_id<0:
            raise ValueError('native transfer requires floating wrists')
        geometry.append(np.c_[q[:,hand.wrist_pos_q_ids],q[:,hand.wrist_orient_q_id:hand.wrist_orient_q_id+4]])
        commands.append(np.c_[target[:,hand.wrist_pos_dof_ids],Rotation.from_rotvec(np.array(target[:,hand.wrist_orient_dof_ids],float,copy=True)).as_quat()])
    geometry=transport_poses(source_object,control_object,np.stack(geometry,axis=1))
    commands=transport_poses(source_object,control_object,np.stack(commands,axis=1))
    new_q=np.array(q,float,copy=True);new_target=np.array(target,float,copy=True)
    for index,hand in enumerate(layout.hands):
        new_q[:,hand.wrist_pos_q_ids]=geometry[:,index,:3]
        new_q[:,hand.wrist_orient_q_id:hand.wrist_orient_q_id+4]=geometry[:,index,3:]
        new_target[:,hand.wrist_pos_dof_ids]=commands[:,index,:3]
        new_target[:,hand.wrist_orient_dof_ids]=Rotation.from_quat(commands[:,index,3:].copy()).as_rotvec()
    unchanged=np.all(np.asarray(source_object)==np.asarray(control_object),axis=1)
    new_q[unchanged]=q[unchanged];new_target[unchanged]=target[unchanged]
    return new_q,new_target


def transfer_named_hand(q,target,source_object,control_object,layout,side):
    """Probe one grasp while leaving every other hand and all fingers exact."""
    hand=layout.hand(side)
    moved_q,moved_target=transfer_floating_reference(q,target,source_object,control_object,layout)
    if np.array_equal(source_object,control_object):return np.array(q,copy=True),np.array(target,copy=True)
    result_q=np.array(q,copy=True);result_target=np.array(target,copy=True)
    qids=tuple(hand.wrist_pos_q_ids)+tuple(range(hand.wrist_orient_q_id,hand.wrist_orient_q_id+4))
    dids=tuple(hand.wrist_pos_dof_ids)+tuple(hand.wrist_orient_dof_ids)
    result_q[:,qids]=moved_q[:,qids];result_target[:,dids]=moved_target[:,dids]
    return result_q,result_target


def _time_arrays(q,target,layout,times):
    q=np.asarray(q);target=np.asarray(target);times=np.asarray(times,float)
    if (q.ndim!=2 or len(q)<2 or target.shape!=(len(q),layout.num_joint_dof) or q.shape[1]!=layout.num_joint_q
            or times.shape!=(len(q),) or not all(np.isfinite(x).all() for x in [q,target,times])
            or times[0]!=0 or times[-1]!=len(q)-1 or np.any(np.diff(times)<=0)):
        raise ValueError('finite ordered frame mapping with fixed endpoints required')
    return q,target,times


def retime_native_fingers(q,target,layout,side,times,*,q_ids=None):
    """Resample selected finger geometry and spring targets without losing preload."""
    q,target,times=_time_arrays(q,target,layout,times)
    hand=layout.hand(side);result_q=q.copy();result_target=target.copy();clock=np.arange(len(q))
    selected=hand.finger_q_ids if q_ids is None else tuple(q_ids)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(hand.finger_q_ids):
        raise ValueError('unique named-hand finger coordinates required')
    mapping=dict(zip(hand.finger_q_ids,hand.finger_dof_ids,strict=True))
    for qid in selected:
        did=mapping[qid]
        result_q[:,qid]=np.interp(times,clock,q[:,qid])
        result_target[:,did]=np.interp(times,clock,target[:,did])
    return result_q,result_target


def retime_native_wrist(q,target,layout,side,times):
    """Retiming both wrist geometry and native commands leaves fingers untouched."""
    q,target,times=_time_arrays(q,target,layout,times);hand=layout.hand(side)
    if hand.wrist_orient_q_id<0 or len(hand.wrist_pos_q_ids)!=3:raise ValueError('floating wrist required')
    result_q=q.copy();result_target=target.copy();clock=np.arange(len(q))
    for qid,did in zip(hand.wrist_pos_q_ids,hand.wrist_pos_dof_ids,strict=True):
        result_q[:,qid]=np.interp(times,clock,q[:,qid]);result_target[:,did]=np.interp(times,clock,target[:,did])
    qi=hand.wrist_orient_q_id;r=hand.wrist_orient_dof_ids
    result_q[:,qi:qi+4]=Slerp(clock,Rotation.from_quat(q[:,qi:qi+4].copy()))(times).as_quat()
    result_target[:,r]=Slerp(clock,Rotation.from_rotvec(target[:,r].copy()))(times).as_rotvec()
    same=times==clock;result_q[same]=q[same];result_target[same]=target[same]
    return result_q,result_target
