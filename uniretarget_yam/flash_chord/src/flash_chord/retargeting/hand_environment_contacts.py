"""Observe solved hand contacts with supports, excluding task objects/hands."""
from dataclasses import dataclass
import warp as wp


@wp.kernel
def aggregate_hand_environment(contact_count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
        shape1: wp.array(dtype=wp.int32), forces: wp.array(dtype=wp.spatial_vector),
        shape_body: wp.array(dtype=wp.int32), shape_hand: wp.array(dtype=wp.int32), body_object: wp.array(dtype=wp.int32),
        shapes_per_world: int, bodies_per_world: int, worlds: int, hands: int,
        force: wp.array(dtype=wp.vec3), intensity: wp.array(dtype=wp.float32),
        counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if i>=contact_count[0]:return
    a=shape0[i];b=shape1[i]
    if a<0 or b<0 or a>=worlds*shapes_per_world or b>=worlds*shapes_per_world:return
    world=a//shapes_per_world
    if b//shapes_per_world!=world:return
    ha=shape_hand[a%shapes_per_world];hb=shape_hand[b%shapes_per_world]
    if (ha>=0)==(hb>=0):return
    hand=ha;other=b;sign=float(1.0)
    if hb>=0:
        hand=hb;other=a;sign=-1.0
    body=shape_body[other]
    if body>=0:
        if body//bodies_per_world!=world:return
        if body_object[body%bodies_per_world]>=0:return
    f=sign*wp.spatial_top(forces[i]);magnitude=wp.length(f);slot=world*hands+hand
    if not wp.isfinite(magnitude):wp.atomic_add(invalid,slot,1)
    elif magnitude>1.0e-4:
        wp.atomic_add(force,slot,f);wp.atomic_add(intensity,slot,magnitude);wp.atomic_add(counts,slot,1)


@dataclass
class HandEnvironmentContacts:
    tracker: object
    shape_body: object
    force: object
    intensity: object
    count: object
    invalid: object

    @classmethod
    def build(cls,tracker,model):
        size=tracker.world_count*len(tracker.layout.sides);device=tracker.shape_hand_ids.device
        return cls(tracker,model.shape_body,wp.zeros(size,dtype=wp.vec3,device=device),
            wp.zeros(size,dtype=wp.float32,device=device),wp.zeros(size,dtype=wp.int32,device=device),wp.zeros(size,dtype=wp.int32,device=device))

    def update(self,contacts):
        for array in [self.force,self.intensity,self.count,self.invalid]:array.zero_()
        t=self.tracker
        if int(contacts.rigid_contact_count.numpy()[0])>contacts.rigid_contact_max:raise ValueError('raw contact overflow invalidates hand-environment feedback')
        if contacts.rigid_contact_max:
            wp.launch(aggregate_hand_environment,dim=contacts.rigid_contact_max,inputs=[contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,contacts.force,self.shape_body,t.shape_hand_ids,t.body_object_ids,
                t.shapes_per_world,t.bodies_per_world,t.world_count,len(t.layout.sides)],outputs=[self.force,self.intensity,self.count,self.invalid],device=self.force.device)
        shape=(t.world_count,len(t.layout.sides))
        return dict(hand_environment_force_on_hand_w=self.force.numpy().reshape(*shape,3).copy(),
            hand_environment_force_norm_sum_n=self.intensity.numpy().reshape(shape).copy(),
            hand_environment_contact_count=self.count.numpy().reshape(shape).copy(),
            hand_environment_invalid_force_count=self.invalid.numpy().reshape(shape).copy())
