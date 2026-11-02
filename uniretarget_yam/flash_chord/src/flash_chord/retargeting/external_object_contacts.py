"""Solved object contacts with non-hand bodies, including static supports."""
from dataclasses import dataclass
import warp as wp


@wp.kernel
def aggregate_external(contact_count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
        shape1: wp.array(dtype=wp.int32), forces: wp.array(dtype=wp.spatial_vector),
        shape_body: wp.array(dtype=wp.int32), shape_hand: wp.array(dtype=wp.int32), body_object: wp.array(dtype=wp.int32),
        shapes_per_world: int, bodies_per_world: int, worlds: int, objects: int,
        force: wp.array(dtype=wp.vec3), magnitude_sum: wp.array(dtype=wp.float32),
        counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if i>=contact_count[0]:return
    a=shape0[i];b=shape1[i]
    if a<0 or b<0 or a>=worlds*shapes_per_world or b>=worlds*shapes_per_world:return
    world=a//shapes_per_world
    if b//shapes_per_world!=world:return
    ba=shape_body[a];bb=shape_body[b];oa=int(-1);ob=int(-1)
    if ba>=0:
        if ba//bodies_per_world!=world:return
        oa=body_object[ba%bodies_per_world]
    if bb>=0:
        if bb//bodies_per_world!=world:return
        ob=body_object[bb%bodies_per_world]
    if oa==ob:return
    ha=shape_hand[a%shapes_per_world];hb=shape_hand[b%shapes_per_world]
    f=wp.spatial_top(forces[i]);length=wp.length(f)
    if oa>=0 and hb<0:
        slot=world*objects+oa
        if not wp.isfinite(length):wp.atomic_add(invalid,slot,1)
        elif length>1.0e-4:
            wp.atomic_add(force,slot,f);wp.atomic_add(magnitude_sum,slot,length);wp.atomic_add(counts,slot,1)
    if ob>=0 and ha<0:
        slot=world*objects+ob
        if not wp.isfinite(length):wp.atomic_add(invalid,slot,1)
        elif length>1.0e-4:
            wp.atomic_add(force,slot,-f);wp.atomic_add(magnitude_sum,slot,length);wp.atomic_add(counts,slot,1)


@dataclass
class ExternalObjectContacts:
    tracker: object
    shape_body: object
    force: object
    magnitude: object
    count: object
    invalid: object

    @classmethod
    def build(cls, tracker, model):
        size=tracker.world_count*tracker.layout.num_objects;device=tracker.shape_hand_ids.device
        return cls(tracker,model.shape_body,wp.zeros(size,dtype=wp.vec3,device=device),
            wp.zeros(size,dtype=wp.float32,device=device),wp.zeros(size,dtype=wp.int32,device=device),wp.zeros(size,dtype=wp.int32,device=device))

    def update(self, contacts):
        for array in [self.force,self.magnitude,self.count,self.invalid]:array.zero_()
        t=self.tracker
        if int(contacts.rigid_contact_count.numpy()[0])>contacts.rigid_contact_max:raise ValueError('raw contact overflow invalidates external support feedback')
        if contacts.rigid_contact_max:
            wp.launch(aggregate_external,dim=contacts.rigid_contact_max,inputs=[contacts.rigid_contact_count,contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
                contacts.force,self.shape_body,t.shape_hand_ids,t.body_object_ids,t.shapes_per_world,t.bodies_per_world,t.world_count,t.layout.num_objects],
                outputs=[self.force,self.magnitude,self.count,self.invalid],device=self.force.device)
        shape=(t.world_count,t.layout.num_objects)
        return dict(object_external_force_on_object_w=self.force.numpy().reshape(*shape,3).copy(),
            object_external_force_norm_sum_n=self.magnitude.numpy().reshape(shape).copy(),
            object_external_contact_count=self.count.numpy().reshape(shape).copy(),
            object_external_invalid_force_count=self.invalid.numpy().reshape(shape).copy())
