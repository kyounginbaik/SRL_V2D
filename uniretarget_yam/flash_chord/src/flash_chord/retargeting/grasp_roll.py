"""Discrete grasp-roll alternatives around the object-to-wrist approach axis.

A family changes contact geometry, not the object task. It must be screened
using fresh physical retention/task outcomes, not its own alignment residual.
"""
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.grasp_feedback import ObjectRelativeGrasp


def roll_grasp(grasp,angle):
    offset=np.asarray(grasp.wrist_position_o,float);length=np.linalg.norm(offset)
    if length<1e-6 or not np.isfinite(angle) or abs(angle)>np.pi+1e-10:
        raise ValueError('a nonzero finite approach offset and a roll within pi are required')
    axis=offset/length;rotation=Rotation.from_rotvec(axis*angle)
    return ObjectRelativeGrasp(tuple(offset),tuple((rotation*Rotation.from_quat(grasp.wrist_quaternion_o)).as_quat()),grasp.fingers)


def roll_replay_specs(specs,hand,bounds):
    result=[];found=False
    for original in specs:
        spec=deepcopy(original)
        if spec['hand']==hand and spec['component'] in ('all','wrist'):
            if spec['component']=='all':result.append(dict(spec,component='fingers'))
            result.append(dict(spec,component='wrist',bounds=list(bounds)));found=True
        else:result.append(spec)
    if not found:raise ValueError('a declared receiving wrist replay is required')
    return result


def roll_wrist_trajectory(wrist,grasp,angle,*,bounds,contact,fps,approach_rate=2.,return_rate=5.):
    base=np.asarray(wrist,float);a,b=bounds;start,end=contact
    if (base.ndim!=2 or base.shape[1]!=7 or not np.isfinite(base).all()
            or not 0<a<start<=end<b<len(base) or not np.isfinite([fps,approach_rate,return_rate]).all()
            or min(fps,approach_rate,return_rate)<=0):
        raise ValueError('finite poses and protected positive-rate grasp transition required')
    rolled=roll_grasp(grasp,angle);rg=Rotation.from_quat(grasp.wrist_quaternion_o)
    local_axis=rg.inv().apply(np.asarray(grasp.wrist_position_o)/np.linalg.norm(grasp.wrist_position_o))
    t=np.arange(len(base));u=np.clip((t-a)/(start-a),0,1);v=np.clip((b-t)/(b-end),0,1)
    smooth=lambda x:x*x*x*(10+x*(-15+6*x))
    weight=smooth(u)*smooth(v)
    rate=np.abs(np.diff(weight))*abs(angle)*fps
    before=float(rate[a:start].max(initial=0));after=float(rate[end:b].max(initial=0))
    if before>approach_rate+1e-6 or after>return_rate+1e-6:
        return None,None,dict(reason='insufficient_full_roll_transition_time',approach_added_rad_s=before,return_added_rad_s=after)
    out=base.copy()
    if angle!=0:out[:,3:]=(Rotation.from_quat(base[:,3:])*Rotation.from_rotvec(weight[:,None]*angle*local_axis)).as_quat()
    out[weight==0]=base[weight==0]
    return out,rolled,dict(reason='full_discrete_grasp_roll',angle_rad=float(angle),bounds=[a,b],contact=[start,end],
        approach_added_rad_s=before,return_added_rad_s=after,semantic_wrist_positions_exact=True,
        contact_angular_velocity_preserved=True)
