"""Read solved inter-hand contact forces, independently of object contact slots."""
from dataclasses import dataclass
import numpy as np
import warp as wp


@wp.kernel
def aggregate_hand_pairs(contact_count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
        shape1: wp.array(dtype=wp.int32), forces: wp.array(dtype=wp.spatial_vector),
        shape_hand: wp.array(dtype=wp.int32), shapes_per_world: int, world_count: int, hand_count: int,
        force_on_hand: wp.array(dtype=wp.vec3), force_norm_sum: wp.array(dtype=wp.float32),
        count: wp.array(dtype=wp.int32), invalid_force_count: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if i>=contact_count[0]:return
    a=shape0[i];b=shape1[i]
    if a<0 or b<0 or a>=world_count*shapes_per_world or b>=world_count*shapes_per_world:return
    world=a//shapes_per_world
    if b//shapes_per_world!=world:return
    h0=shape_hand[a%shapes_per_world];h1=shape_hand[b%shapes_per_world]
    if h0<0 or h1<0 or h0==h1:return
    f=wp.spatial_top(forces[i]);magnitude=wp.length(f)
    p0=world*hand_count+h0;p1=world*hand_count+h1
    if not wp.isfinite(magnitude):
        wp.atomic_add(invalid_force_count,p0,1);wp.atomic_add(invalid_force_count,p1,1)
        return
    if magnitude<=1.0e-4:return
    # Native contact force acts on shape0; shape1 receives its negative.
    wp.atomic_add(force_on_hand,p0,f);wp.atomic_add(force_on_hand,p1,-f)
    wp.atomic_add(force_norm_sum,p0,magnitude);wp.atomic_add(force_norm_sum,p1,magnitude)
    wp.atomic_add(count,p0,1);wp.atomic_add(count,p1,1)


@dataclass
class HandPairContacts:
    tracker: object
    force: object
    intensity: object
    count: object
    invalid: object

    @classmethod
    def build(cls, tracker):
        n=tracker.world_count*len(tracker.layout.sides);device=tracker.shape_hand_ids.device
        return cls(tracker,wp.zeros(n,dtype=wp.vec3,device=device),
            wp.zeros(n,dtype=wp.float32,device=device),wp.zeros(n,dtype=wp.int32,device=device),wp.zeros(n,dtype=wp.int32,device=device))

    def update(self, contacts):
        self.force.zero_();self.intensity.zero_();self.count.zero_();self.invalid.zero_()
        t=self.tracker
        if int(contacts.rigid_contact_count.numpy()[0])>contacts.rigid_contact_max:
            raise ValueError('raw contact overflow cannot certify hand-pair feedback')
        if contacts.rigid_contact_max:
            wp.launch(aggregate_hand_pairs,dim=contacts.rigid_contact_max,inputs=[contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,contacts.force,t.shape_hand_ids,
                t.shapes_per_world,t.world_count,len(t.layout.sides)],outputs=[self.force,self.intensity,self.count,self.invalid],device=self.force.device)
        shape=(t.world_count,len(t.layout.sides))
        return dict(hand_hand_force_on_hand_w=self.force.numpy().reshape(*shape,3).copy(),
            hand_hand_force_norm_sum_n=self.intensity.numpy().reshape(shape).copy(),
            hand_hand_contact_count=self.count.numpy().reshape(shape).copy(),
            hand_hand_invalid_force_count=self.invalid.numpy().reshape(shape).copy())
