"""Replay captured robot actuator targets through ordinary residual actions.

No simulation state or action buffer is written. The caller executes the
returned policy input normally and observes its processed targets afterwards.
This diagnostic controller is not an exported learned policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_feedback import encode_filtered_residual, hand_target_residual


def absorb_finger_commands(reference_q, reference_target, commands, q_ids, dof_ids,
                            lower, upper, *, bounds, fps, speed_rad_s=3.):
    """Absorb the physical part of a command into a bounded finger reference.

    The remaining command-minus-reference is deliberately retained as actuator
    preload. This function never clips or rewrites the captured commands.
    Both influence boundaries, other channels and trajectory endpoints stay
    exact. The ramp bound describes its envelope, not noisy command-tape slew.
    """
    base=np.asarray(reference_q);target=np.asarray(reference_target);command=np.asarray(commands)
    ids=np.asarray(q_ids);dids=np.asarray(dof_ids);lo=np.asarray(lower);hi=np.asarray(upper)
    if (base.ndim!=2 or target.ndim!=2 or len(base)!=len(target) or command.shape!=(len(base)-1,target.shape[1])
            or ids.ndim!=1 or not len(ids) or dids.shape!=ids.shape or lo.shape!=ids.shape or hi.shape!=ids.shape
            or not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(dids.dtype,np.integer)
            or len(set(ids))!=len(ids) or len(set(dids))!=len(dids)
            or ids.min()<0 or ids.max()>=base.shape[1] or dids.min()<0 or dids.max()>=target.shape[1]
            or not all(np.isfinite(x).all() for x in (base,target,command,lo,hi)) or np.any(lo>hi)
            or not np.isfinite([fps,speed_rad_s]).all() or min(fps,speed_rad_s)<=0):
        raise ValueError('finite aligned references, actuator commands and named finger limits required')
    a,b=bounds
    if any(not isinstance(v,(int,np.integer)) for v in (a,b)) or not 1<=a<b<len(command):
        raise ValueError('protected replay boundaries must lie after reset and within executed frames')
    if np.any(base[:,ids]<lo-1e-6) or np.any(base[:,ids]>hi+1e-6):raise ValueError('source reference violates joint limits')
    bounded=np.clip(command[:,dids],lo,hi)
    peak=float(np.abs(bounded[a:b+1]-base[a:b+1][:,ids]).max())
    ramp=max(1,int(np.ceil(1.875*peak*fps/speed_rad_s)))
    if 2*ramp>b-a:raise ValueError('insufficient approach/restore time for command absorption')
    t=np.arange(len(command));rise=np.clip((t-a)/ramp,0,1);fall=np.clip((b-t)/ramp,0,1)
    smooth=lambda x:x*x*x*(10+x*(-15+6*x))
    weight=smooth(rise)*smooth(fall)
    q=np.array(base,dtype=float,copy=True);nt=np.array(target,copy=True)
    q[:-1,ids]=base[:-1,ids]+weight[:,None]*(bounded-base[:-1,ids])
    unchanged=weight==0;q[np.flatnonzero(unchanged)]=base[np.flatnonzero(unchanged)]
    nt[:-1,dids]=q[:-1,ids];nt[np.flatnonzero(unchanged)]=target[np.flatnonzero(unchanged)]
    residual=command[:,dids]-q[:-1,ids]
    return q,nt,dict(bounds=[int(a),int(b)],ramp_frames=ramp,
        command_preload_max_rad=float(np.abs(residual[a:b+1]).max()),
        source_command_max_outside_physical_limits_rad=float(np.maximum(np.maximum(command[:,dids]-hi,lo-command[:,dids]),0).max()),
        command_targets_unchanged=True)


class CommandReplay:
    def __init__(self, targets, groups, candidates):
        self.targets=np.asarray(targets)
        self.groups=np.asarray(groups)
        self.candidates=candidates
        if (self.targets.ndim!=3 or self.targets.dtype.kind!='f' or not np.isfinite(self.targets).all()
                or self.groups.ndim!=1 or not np.issubdtype(self.groups.dtype,np.integer)
                or np.any(self.groups<0) or np.any(self.groups>=len(candidates))
                or len(candidates)!=len(self.targets)):
            raise ValueError('finite [candidate,frame,robot DOF] targets and valid groups required')
        for candidate in candidates:
            for spec in candidate.get('command_replay',[]):
                a,b=spec['bounds']
                if (any(not isinstance(v,(int,np.integer)) for v in (a,b))
                        or not 1<=a<=b<self.targets.shape[1]
                        or spec['component'] not in ('all','wrist','fingers')):
                    raise ValueError('replay requires an inclusive executed interval starting after frame zero')
        self.env=None;self.errors=[];self.saturation=[]

    def _bind(self,env):
        if env.world_count!=len(self.groups):raise ValueError('replay world population changed')
        delay=getattr(env.action,'delay',None)
        if delay is not None and delay.config.max_steps!=0:
            raise ValueError('command replay currently requires zero actuator delay')
        buffers=env.action.buffers;layout=buffers.layout;ref=buffers.reference
        if tuple(ref.sides)!=tuple(layout.sides):raise ValueError('reference/action hand ordering differs')
        self.frames=ref.num_frames;self.action_layout=layout
        if self.targets.shape[1]!=self.frames-1:raise ValueError('replay command/reference horizon differs')
        rw=ref.reference_world_count
        if rw not in (1,env.world_count):raise ValueError('unsupported reference population')
        def expand(x,tail):
            values=x.numpy().reshape(self.frames,rw,*tail)
            return np.broadcast_to(values,(self.frames,env.world_count,*tail))
        self.position=expand(ref.wrist_pos_w,(layout.num_hands,3))
        self.quaternion=expand(ref.wrist_quat_w,(layout.num_hands,4))
        self.fingers=expand(ref.finger_joint_pos,(layout.total_fingers,))
        self.scale=buffers.scale.numpy().copy();self.clip=buffers.clip.numpy().copy()
        self.input_scale=buffers.input_scale.numpy().copy();self.mapping=buffers.input_mapping;self.ema=buffers.ema
        self.specs=[]
        for world,group in enumerate(self.groups):
            for spec in self.candidates[int(group)].get('command_replay',[]):
                if spec['hand'] not in layout.sides:raise ValueError('unknown replay hand')
                h=layout.sides.index(spec['hand']);start=layout.hand_action_starts[h];n=layout.finger_counts[h]
                qlayout=env.scene.layout.hand(spec['hand'])
                native=(*qlayout.wrist_pos_dof_ids,*qlayout.wrist_orient_dof_ids,*qlayout.finger_dof_ids)
                if len(native)!=6+n or max(native)>=self.targets.shape[-1]:raise ValueError('replay target DOFs do not match floating hand')
                component=spec['component']
                local=np.arange(6+n) if component=='all' else (np.arange(6) if component=='wrist' else np.arange(6,6+n))
                self.specs.append((world,int(group),h,start,n,np.asarray(native),local,tuple(spec['bounds'])))
        self.env=env

    def action(self, action, frame, env):
        if env is None:raise ValueError('replay requires the live previous-step environment')
        if self.env is not env:self._bind(env)
        if not 1<=frame<self.targets.shape[1]:raise ValueError('replay frame outside executed horizon')
        if not np.array_equal(env.timestep.numpy(),np.full(env.world_count,frame)):
            raise ValueError('replay is not at the pre-action frame boundary')
        out=np.array(action,copy=True)
        previous=env.action.filtered_action.numpy().reshape(env.world_count,-1)
        if out.shape!=previous.shape:raise ValueError('policy action shape mismatch')
        for world,group,h,start,n,native,local,(a,b) in self.specs:
            if not a<=frame<=b:continue
            target=self.targets[group,frame,native]
            wrist=np.r_[target[:3],Rotation.from_rotvec(target[3:6]).as_quat()]
            finger_start=self.action_layout.hand_finger_starts[h]
            ref_wrist=np.r_[self.position[frame,world,h],self.quaternion[frame,world,h]]
            reference_fingers=self.fingers[frame,world,finger_start:finger_start+n]
            desired=hand_target_residual(ref_wrist,reference_fingers,wrist,target[6:])
            ids=start+local
            encoded=encode_filtered_residual(desired[local],previous[world,ids],self.scale[ids],self.clip[ids],
                ema=self.ema,mapping=self.mapping,input_scale=self.input_scale[ids])
            out[world,ids]=encoded.action
            self.saturation.append((frame,world,float(encoded.saturated.mean())))
        return out

    def observe(self,env,frame):
        if self.env is not env:return
        actual=env.action.joint_target.numpy().reshape(env.world_count,-1)
        for world,group,h,start,n,native,local,(a,b) in self.specs:
            if not a<=frame<=b:continue
            desired=self.targets[group,frame,native];observed=actual[world,native]
            p=float(np.linalg.norm(observed[:3]-desired[:3])) if np.any(local<3) else 0.
            r=float((Rotation.from_rotvec(observed[3:6])*Rotation.from_rotvec(desired[3:6]).inv()).magnitude()) if np.any((local>=3)&(local<6)) else 0.
            f=float(np.abs(observed[6:]-desired[6:]).max()) if np.any(local>=6) else 0.
            self.errors.append((frame,world,p,r,f))

    def arrays(self):
        return dict(command_replay_errors=np.asarray(self.errors,float).reshape(-1,5),
                    command_replay_saturation=np.asarray(self.saturation,float).reshape(-1,3))
