"""Gated unused-finger actor with independently adjustable safe pinky posture."""
import numpy as np
from posture_feedback_policy import PosturePolicy as BasePosturePolicy


class PosturePolicy(BasePosturePolicy):
    def __call__(self, next_time_s, qpos, qvel):
        output = super().__call__(next_time_s, qpos, qvel)
        gate = self.gate(next_time_s)
        if gate > 0:
            ids = self.data['posture_actuator_ids']
            output[ids] += gate * self.data.get('posture_joint_bias_rad', 0.)
            output = np.clip(output, self.data['ctrl_lo'], self.data['ctrl_hi'])
        return output
