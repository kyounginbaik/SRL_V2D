"""Install causal HAND references before ordinary residual action processing.

Only private action-reference arrays change. Geometry observations/rewards,
objects, physical state, processed targets, filters, gains and force limits do
not. This diagnostic adapter is not an exported learned policy.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from .causal_grasp_reference import CausalGraspReference
from .pose_response import PoseResponse
from .position_response import PositionResponse


@wp.kernel
def scatter_reference_rows(frames: wp.array(dtype=wp.int32), hands: wp.array(dtype=wp.int32),
        starts: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
        worlds: int, hand_count: int, total_fingers: int,
        position: wp.array(dtype=wp.vec3), rotation: wp.array(dtype=wp.quat),
        fingers: wp.array(dtype=wp.float32),
        ref_position: wp.array(dtype=wp.vec3), ref_rotation: wp.array(dtype=wp.quat),
        ref_fingers: wp.array(dtype=wp.float32)):
    world = wp.tid(); hand = hands[world]
    if hand >= 0:
        row = frames[world]*worlds + world
        ref_position[row*hand_count+hand] = position[world]
        ref_rotation[row*hand_count+hand] = rotation[world]
        first = starts[hand]
        for j in range(counts[hand]):
            ref_fingers[row*total_fingers+first+j] = fingers[world*total_fingers+first+j]


class CausalReferenceInstaller:
    def __init__(self, goals, objects, initial_reference_object0, targets, groups, candidates, fps,
                 *, array_factory=None, row_updater=None):
        self.goals=np.asarray(goals);self.objects=np.asarray(objects);self.reference0=initial_reference_object0
        self.targets=np.asarray(targets);self.groups=np.asarray(groups);self.candidates=candidates;self.fps=fps
        self.array_factory=array_factory;self.row_updater=row_updater;self.instances=[]
        if (self.goals.ndim!=3 or self.goals.shape[-1]!=7 or self.objects.shape!=(self.goals.shape[1],7)
                or self.targets.ndim!=3 or self.targets.shape[:2]!=self.goals.shape[:2]
                or len(candidates)!=len(self.goals) or self.groups.ndim!=1
                or not np.issubdtype(self.groups.dtype,np.integer) or np.any(self.groups<0) or np.any(self.groups>=len(candidates))
                or not all(np.isfinite(x).all() for x in [self.goals,self.objects,self.targets])):
            raise ValueError('finite candidate geometry/nominals, task path and world groups required')

    def __call__(self, env):
        if env.world_count!=len(self.groups) or np.any(env.timestep.numpy()!=0):
            raise ValueError('causal references require a fresh matching population')
        run=_CausalReferenceRun(self,env);self.instances.append(run)
        original=env.action.process
        def process(action,timestep,state=None):
            run.before_action(timestep,state)
            return original(action,timestep,state)
        env.action.process=process


class _CausalReferenceRun:
    def __init__(self, owner, env):
        self.owner=owner;self.env=env;self.records=[];self.disabled=set();self.invalid=[]
        ref=env.action.buffers.reference;layout=env.action.buffers.layout;w=env.world_count
        if ref.reference_world_count!=w or ref.num_frames-1!=owner.goals.shape[1]:
            raise ValueError('causal references require packed matching world trajectories')
        self.layout=layout;self.base_ref=ref;self.controllers={};self.hand_ids=np.full(w,-1,np.int32)
        self.object_ids=env.command.body_ids_w.numpy().reshape(w,-1)
        self.body_count=env.command.bodies_per_world
        self.palm_frames={};self.native_fingers={}
        for world,group in enumerate(owner.groups):
            specification=owner.candidates[int(group)].get('causal_grasp_reference')
            if specification is None:continue
            spec=dict(specification);side=spec.pop('hand');obj=spec.pop('object_id')
            if not 0<=obj<self.object_ids.shape[1] or side not in layout.sides:raise ValueError('unknown controlled hand/object')
            hand=env.scene.layout.hand(side);self.palm_frames[world]=hand.palm_frame
            self.native_fingers[world]=list(hand.finger_dof_ids);self.hand_ids[world]=layout.sides.index(side)
            model=PoseResponse(**spec.pop('model')) if 'model' in spec else None
            position=PositionResponse(**spec.pop('position_model')) if 'position_model' in spec else None
            nominal=np.c_[owner.targets[group][:,hand.wrist_pos_dof_ids],
                Rotation.from_rotvec(owner.targets[group][:,hand.wrist_orient_dof_ids].copy()).as_quat()]
            self.controllers[world]=(obj,CausalGraspReference(owner.goals[group],owner.objects,nominal,
                owner.targets[group][:,hand.finger_dof_ids],initial_reference_object0=owner.reference0,
                fps=owner.fps,model=model,position_model=position,**spec))
        def array(values,old):
            if owner.array_factory:return owner.array_factory(values,old)
            return wp.array(values,dtype=old.dtype,device=old.device)
        # Give action-owned nominal inputs independent storage. The source
        # geometry may be shared by observations/rewards and must stay exact.
        env.action.buffers.reference=replace(ref,
            wrist_pos_w=array(ref.wrist_pos_w.numpy().copy(),ref.wrist_pos_w),
            wrist_quat_w=array(ref.wrist_quat_w.numpy().copy(),ref.wrist_quat_w),
            finger_joint_pos=array(ref.finger_joint_pos.numpy().copy(),ref.finger_joint_pos))
        self.base_p=ref.wrist_pos_w.numpy().reshape(ref.num_frames,w,len(layout.sides),3).copy()
        self.base_r=ref.wrist_quat_w.numpy().reshape(ref.num_frames,w,len(layout.sides),4).copy()
        self.base_f=ref.finger_joint_pos.numpy().reshape(ref.num_frames,w,layout.total_fingers).copy()
        self.initial0={}

    def before_action(self,timestep,state):
        env=self.env;w=env.world_count;frames=np.asarray(timestep.numpy())
        if frames.shape!=(w,) or np.any(frames<0) or np.any(frames>=self.base_ref.num_frames-1):
            raise ValueError('valid per-world pre-action reference frames required')
        if state is None:raise ValueError('causal references need measured BEFORE state')
        body=state.body_q.numpy();native=env.action.joint_target.numpy().reshape(w,-1)
        position=np.zeros((w,3),np.float32);rotation=np.zeros((w,4),np.float32);rotation[:,3]=1
        fingers=np.zeros((w,self.layout.total_fingers),np.float32)
        goals=np.zeros((w,7));nominals=goals.copy();mask=self.hand_ids>=0
        for world,(obj,controller) in self.controllers.items():
            frame=int(frames[world]);h=self.hand_ids[world];first=self.layout.hand_finger_starts[h];count=self.layout.finger_counts[h]
            if frame==0:self.disabled.discard(world)
            pose=body[self.object_ids[world,obj]];initial=body[self.object_ids[world,0]]
            palm=self.palm_frames[world];bp=body[world*self.body_count+palm.body_id]
            valid=all(np.isfinite(x).all() and np.linalg.norm(x[3:])>1e-8 for x in [pose,initial,bp])
            if not valid:self.disabled.add(world);self.invalid.append((len(self.records),world))
            if world in self.disabled:
                nominal=np.r_[self.base_p[frame,world,h],self.base_r[frame,world,h]];finger=self.base_f[frame,world,first:first+count];goal=nominal
            else:
                br=Rotation.from_quat(bp[3:]);wrist=np.r_[bp[:3]+br.apply(palm.body_to_frame_pos),(br*Rotation.from_quat(palm.body_to_frame_quat_xyzw)).as_quat()]
                nominal,finger,info=controller.step(frame,pose,wrist,initial,native[world,self.native_fingers[world]])
                goal=info['goal']
            position[world]=nominal[:3];rotation[world]=nominal[3:];fingers[world,first:first+count]=finger
            goals[world]=goal;nominals[world]=nominal
        ref=env.action.buffers.reference
        if self.owner.row_updater:
            self.owner.row_updater(ref,frames,self.hand_ids,position,rotation,fingers,self.layout)
        else:
            device=ref.wrist_pos_w.device
            inputs=[wp.array(frames,dtype=wp.int32,device=device),wp.array(self.hand_ids,dtype=wp.int32,device=device),
                wp.array(self.layout.hand_finger_starts,dtype=wp.int32,device=device),wp.array(self.layout.finger_counts,dtype=wp.int32,device=device),
                w,len(self.layout.sides),self.layout.total_fingers,wp.array(position,dtype=wp.vec3,device=device),
                wp.array(rotation,dtype=wp.quat,device=device),wp.array(fingers.reshape(-1),dtype=wp.float32,device=device)]
            wp.launch(scatter_reference_rows,dim=w,inputs=inputs,outputs=[ref.wrist_pos_w,ref.wrist_quat_w,ref.finger_joint_pos],device=device)
        self.records.append((goals,nominals,mask.copy(),fingers))

    def arrays(self):
        return dict(causal_wrist_goal=np.array([x[0] for x in self.records]),
            causal_wrist_nominal=np.array([x[1] for x in self.records]),causal_controlled_world=self.hand_ids>=0,
            causal_finger_nominal=np.array([x[3] for x in self.records]),
            causal_seated_frame=np.array([self.controllers[w][1].seated_at if w in self.controllers else -1 for w in range(self.env.world_count)]),
            causal_closure_command_frame=np.array([self.controllers[w][1].closed_at if w in self.controllers else -1 for w in range(self.env.world_count)]),
            causal_invalid_samples=np.asarray(self.invalid,dtype=int).reshape(-1,2))
