"""Independent complete hand-contact recording for physical-response diagnosis."""
import ast

import numpy as np


class ContactFeedbackRecorder:
    def __init__(self, factory=None, *, record_execution=False, record_hand_pairs=False, record_external_contacts=False):
        self.factory=factory;self.env=None;self.record_execution=record_execution;self.record_hand_pairs=record_hand_pairs
        self.record_external_contacts=record_external_contacts

    def __call__(self, env, frame):
        if self.env is not env:
            if frame!=0:raise ValueError('contact recording must begin at executed frame zero')
            if self.factory is None:
                from .complete_contacts import build_complete_hand_contacts
                factory=build_complete_hand_contacts
            else:factory=self.factory
            self.observer=factory(env);self.env=env
            self.sides=tuple(self.observer.sides)
            self.slot_hand=np.asarray(self.observer.slot_hand);self.slot_object=np.asarray(self.observer.slot_object)
            ids=env.command.body_ids_w.numpy().reshape(env.world_count,-1)
            self.mass=np.asarray(env.model.body_mass.numpy())[ids]
            self.gravity=np.broadcast_to(env.model.gravity.numpy().reshape(-1,3),(env.world_count,3)).copy()
            self.forces=[];self.counts=[]
            if self.record_hand_pairs:
                from .hand_pair_contacts import HandPairContacts
                self.hand_pairs=HandPairContacts.build(self.observer.tracker);self.hand_pair_records=[]
            if self.record_external_contacts:
                from .external_object_contacts import ExternalObjectContacts
                self.external=ExternalObjectContacts.build(self.observer.tracker,env.model);self.external_records=[]
                from .hand_environment_contacts import HandEnvironmentContacts
                self.hand_environment=HandEnvironmentContacts.build(self.observer.tracker,env.model);self.hand_environment_records=[]
            if self.record_execution:
                self.robot_nq=int(env.scene.robot_reference.num_joint_q)
                self.robot_nd=int(env.scene.robot_reference.num_joint_dof)
                if min(self.robot_nq,self.robot_nd)<=0:raise ValueError('positive robot execution dimensions required')
                self.executed_q=[];self.commanded_target=[];self.applied_target=[]
                self.link_forces=[];self.link_active=[]
                self.slot_links=tuple(getattr(self.observer,'slot_links',tuple(f'slot_{i}' for i in range(len(self.slot_hand)))))
                if len(self.slot_links)!=len(self.slot_hand):raise ValueError('physical link names must match contact slots')
        if frame!=len(self.forces):raise ValueError('contact recording frames must be contiguous')
        data=self.observer.update(env)
        force=np.asarray(data['force'],float);active=np.asarray(data['active'])>0
        if force.shape!=(*active.shape,3):raise ValueError('contact force/active schema mismatch')
        force=np.where(active[...,None],force,0.)
        values=np.zeros((env.world_count,len(self.sides),self.mass.shape[1],3),np.float32)
        counts=np.zeros(values.shape[:-1],np.int32)
        for h in range(len(self.sides)):
            for o in range(self.mass.shape[1]):
                slots=(self.slot_hand==h)&(self.slot_object==o)
                values[:,h,o]=force[:,slots].sum(1)
                counts[:,h,o]=(active[:,slots]&(np.linalg.norm(force[:,slots],axis=-1)>1e-4)).sum(1)
        self.forces.append(values);self.counts.append(counts)
        if self.record_hand_pairs:self.hand_pair_records.append(self.hand_pairs.update(env.contacts))
        if self.record_external_contacts:
            self.external_records.append(self.external.update(env.contacts))
            self.hand_environment_records.append(self.hand_environment.update(env.contacts))
        if self.record_execution:
            self.link_forces.append(force.astype(np.float32,copy=True));self.link_active.append(active.copy())
            q=env.state_0.joint_q.numpy().reshape(env.world_count,-1)
            command=env.action.joint_target.numpy().reshape(env.world_count,-1)
            delay=getattr(env.action,'delay',None)
            applied=command if delay is None else delay.delayed_target.numpy().reshape(env.world_count,-1)
            if q.shape[1]<self.robot_nq or min(command.shape[1],applied.shape[1])<self.robot_nd:
                raise ValueError('native execution buffers omit robot coordinates')
            self.executed_q.append(q[:,:self.robot_nq].copy())
            self.commanded_target.append(command[:,:self.robot_nd].copy())
            self.applied_target.append(applied[:,:self.robot_nd].copy())

    def arrays(self):
        if self.env is None:raise ValueError('no physical contacts were observed')
        forces=np.asarray(self.forces);g=np.linalg.norm(self.gravity,axis=-1)
        up=-self.gravity/np.maximum(g[:,None],1e-12)
        load=np.einsum('fwhoj,wj->fwho',forces,up)/(np.maximum(self.mass*g[:,None],1e-12)[None,:,None,:])
        result=dict(hand_object_force_w=forces,hand_object_contact_count=np.asarray(self.counts),
            hand_object_support_mg=load,physical_contact_sides=np.asarray(self.sides),
            object_mass_kg=self.mass,gravity_w=self.gravity,contact_frame_ids=np.arange(len(forces)))
        if self.record_hand_pairs:
            for key in self.hand_pair_records[0]:result[key]=np.asarray([row[key] for row in self.hand_pair_records])
        if self.record_external_contacts:
            for key in self.external_records[0]:result[key]=np.asarray([row[key] for row in self.external_records])
            for key in self.hand_environment_records[0]:result[key]=np.asarray([row[key] for row in self.hand_environment_records])
        if self.record_execution:
            result.update(executed_robot_joint_q=np.asarray(self.executed_q),
                complete_link_force_on_object_w=np.asarray(self.link_forces),complete_link_contact_active=np.asarray(self.link_active),
                complete_link_names=np.asarray(self.slot_links),complete_slot_hand_ids=self.slot_hand.copy(),complete_slot_object_ids=self.slot_object.copy(),
                commanded_robot_joint_target=np.asarray(self.commanded_target),
                applied_robot_joint_target=np.asarray(self.applied_target),
                execution_frame_ids=np.arange(len(forces)),
                applied_target_semantics=np.asarray('last physics substep of the recorded control frame'))
        return result


def patch_rollout_text(source):
    marker='# PHYSICAL_RESPONSE_OBSERVER_V1'
    if marker in source:return source
    anchor="        jax.block_until_ready(obs);_clock['policy_and_sim']+=_time.perf_counter()-_t;_t=_time.perf_counter()\n"
    if source.count(anchor)!=1:raise ValueError('unsupported rollout: expected one post-step observer insertion point')
    text=source.replace(anchor,anchor+'''        # PHYSICAL_RESPONSE_OBSERVER_V1
        _physical_observer=getattr(editor,'physical_response_observer',None)
        if _physical_observer is not None: _physical_observer(env,frame)
''',1)
    ast.parse(text)
    return text
