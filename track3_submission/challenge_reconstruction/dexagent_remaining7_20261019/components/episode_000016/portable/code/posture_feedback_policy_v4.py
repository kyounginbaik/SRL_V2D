"""Moderate unused-finger curl with a bounded, post-pickup grasp countercorrection."""
import numpy as np
from scipy.spatial.transform import Rotation
from posture_feedback_policy_v3 import PosturePolicy as ParentPolicy


class PosturePolicy(ParentPolicy):
    def __call__(self, next_time_s, qpos, qvel):
        output = super().__call__(next_time_s, qpos, qvel)
        gate = self.gate(next_time_s)
        if gate <= 0:
            return output
        d = self.data
        ids = d['grasp_wrist_actuator_ids']
        original = output[ids[3:]].copy()
        rotation = Rotation.from_euler('XYZ', original)
        output[ids[:3]] += rotation.apply(gate*d['grasp_translation_local_m'])
        # A zero correction must be bitwise the parent actor, avoiding a
        # gratuitous Euler round trip in the initial reproducibility check.
        if np.any(d['grasp_rotation_local_rad'] != 0):
            desired = rotation*Rotation.from_rotvec(gate*d['grasp_rotation_local_rad'])
            e = desired.as_euler('XYZ')
            candidates = np.array([e, [e[0]+np.pi, np.pi-e[1], e[2]+np.pi]])
            candidates += 2*np.pi*np.round((original-candidates)/(2*np.pi))
            output[ids[3:]] = candidates[np.argmin(np.sum((candidates-original)**2,axis=1))]
        output[d['grasp_flex_actuator_ids']] += gate*d['grasp_flex_bias_rad']
        return np.clip(output,d['ctrl_lo'],d['ctrl_hi'])
