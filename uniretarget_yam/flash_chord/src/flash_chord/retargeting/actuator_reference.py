"""Bind explicit wrist spring setpoints separately from geometric references.

This is setup-time reference wiring, before graph capture and the first step.
Geometry used by observation/reward, finger targets, task commands, actuator
gains and simulation state remain untouched. Normal action processing, residual
limits, target delay and wrist effort limits still apply afterwards.
"""
import ast
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation


class WristNominalReference:
    def __init__(self,targets,groups,candidates,*,array_factory=None):
        self.targets=np.asarray(targets);self.groups=np.asarray(groups);self.candidates=candidates
        self.array_factory=array_factory;self.installations=[]
        if (self.targets.ndim!=3 or self.targets.dtype.kind!='f' or not np.isfinite(self.targets).all()
                or self.groups.ndim!=1 or not np.issubdtype(self.groups.dtype,np.integer)
                or len(candidates)!=len(self.targets) or np.any(self.groups<0) or np.any(self.groups>=len(candidates))):
            raise ValueError('finite [candidate,frame,DOF] nominal setpoints and valid world groups required')
        for candidate in candidates:
            for spec in candidate.get('wrist_nominal_reference',[]):
                a,b=spec['bounds']
                if any(not isinstance(v,(int,np.integer)) for v in (a,b)) or not 1<=a<=b<self.targets.shape[1]:
                    raise ValueError('nominal wrist influence must protect reset and final reference rows')

    def __call__(self,env):
        if env.world_count!=len(self.groups) or np.any(env.timestep.numpy()!=0):
            raise ValueError('nominal references must be installed in their fresh population before stepping')
        ref=env.action.buffers.reference;frames=ref.num_frames;sides=tuple(ref.sides)
        if self.targets.shape[1]!=frames-1 or getattr(ref,'reference_world_count',1)!=env.world_count:
            raise ValueError('nominal setpoints require matching, already packed world references')
        p=ref.wrist_pos_w.numpy().reshape(frames,env.world_count,len(sides),3).copy()
        r=ref.wrist_quat_w.numpy().reshape(frames,env.world_count,len(sides),4).copy()
        original_p=p.copy();original_r=r.copy();controlled=0
        for world,group in enumerate(self.groups):
            for spec in self.candidates[int(group)].get('wrist_nominal_reference',[]):
                if spec['hand'] not in sides:raise ValueError('unknown nominal wrist hand')
                hand=env.scene.layout.hand(spec['hand']);h=sides.index(spec['hand']);a,b=spec['bounds']
                pos=tuple(hand.wrist_pos_dof_ids);ori=tuple(hand.wrist_orient_dof_ids)
                if len(pos)!=3 or len(ori)!=3 or max((*pos,*ori))>=self.targets.shape[-1]:
                    raise ValueError('native floating-wrist setpoint schema required')
                p[a:b+1,world,h]=self.targets[group,a:b+1][:,pos]
                r[a:b+1,world,h]=Rotation.from_rotvec(self.targets[group,a:b+1][:,ori]).as_quat()
                controlled+=b-a+1
        def array(values,old):
            if self.array_factory is not None:return self.array_factory(values,old)
            import warp as wp
            return wp.array(values,dtype=old.dtype,device=old.device)
        # Rebind only action-owned nominal wrist fields. No processed target,
        # filtered action, geometry observation or native state is assigned.
        env.action.buffers.reference=replace(ref,
            wrist_pos_w=array(p.reshape(-1,3),ref.wrist_pos_w),
            wrist_quat_w=array(r.reshape(-1,4),ref.wrist_quat_w))
        change=np.linalg.norm(p-original_p,axis=-1)
        angle=(Rotation.from_quat(r.reshape(-1,4))*Rotation.from_quat(original_r.reshape(-1,4)).inv()).magnitude()
        self.installations.append(dict(controlled_hand_frames=controlled,
            nominal_position_change_max_m=float(change.max(initial=0)),
            nominal_orientation_change_max_rad=float(angle.max(initial=0)),
            geometric_reference_unchanged=True,finger_reference_unchanged=True))


def patch_nominal_reference_hook(source):
    marker='# ACTUATOR_WRIST_NOMINAL_REFERENCE_V1'
    if marker in source:return source
    anchor='    packed=pack_candidate_references(env,q,targets);install_candidate_references(env,packed)\n'
    if source.count(anchor)!=1:raise ValueError('unsupported rollout nominal-reference insertion point')
    patched=source.replace(anchor,anchor+'''    # ACTUATOR_WRIST_NOMINAL_REFERENCE_V1
    _nominal_installer=getattr(editor,'actuator_reference_installer',None)
    if _nominal_installer is not None: _nominal_installer(env)
''',1)
    ast.parse(patched);return patched
