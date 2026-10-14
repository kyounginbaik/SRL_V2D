"""Optional physical speed caps for floating wrists, applied after each substep.

These are velocity projections, not certified collision avoidance. FK must be
refreshed by the caller so body velocities agree with the capped joint state.
"""
from dataclasses import dataclass
import math
import warp as wp


@wp.kernel
def _cap_wrist_velocity(qd: wp.array(dtype=wp.float32), ids: wp.array(dtype=wp.int32),
                        dofs_per_world: int, linear_limit: float, angular_limit: float):
    world, hand = wp.tid()
    for block in range(2):
        offset = hand * 6 + block * 3
        i = world * dofs_per_world + ids[offset]
        j = world * dofs_per_world + ids[offset + 1]
        k = world * dofs_per_world + ids[offset + 2]
        velocity = wp.vec3(qd[i], qd[j], qd[k])
        limit = linear_limit
        if block == 1:
            limit = angular_limit
        speed = wp.length(velocity)
        if speed > limit:
            velocity = velocity * (limit / speed)
            qd[i] = velocity[0]
            qd[j] = velocity[1]
            qd[k] = velocity[2]


@dataclass
class FloatingWristVelocityLimits:
    world_count: int
    hand_count: int
    dofs_per_world: int
    ids: wp.array
    linear_limit: float
    angular_limit: float

    @classmethod
    def build(cls, model, layout, world_count, linear_limit, angular_limit, device=None):
        if not all(math.isfinite(x) and x > 0 for x in (linear_limit, angular_limit)):
            raise ValueError('Both floating wrist speed limits must be finite and positive')
        ids = []
        for hand in layout.hands:
            if len(hand.wrist_pos_dof_ids) != 3 or len(hand.wrist_orient_dof_ids) != 3:
                raise ValueError('Wrist speed caps require floating 3-translation + ball wrists')
            ids.extend((*hand.wrist_pos_dof_ids, *hand.wrist_orient_dof_ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Expected distinct floating wrist DOFs')
        return cls(world_count, len(layout.hands), model.joint_dof_count // world_count,
                   wp.array(ids, dtype=wp.int32, device=device or model.device),
                   float(linear_limit), float(angular_limit))

    def enforce_velocity(self, state):
        wp.launch(_cap_wrist_velocity, dim=(self.world_count, self.hand_count),
                  inputs=[state.joint_qd, self.ids, self.dofs_per_world,
                          self.linear_limit, self.angular_limit], device=state.joint_qd.device)
