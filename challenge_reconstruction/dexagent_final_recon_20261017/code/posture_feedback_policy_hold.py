"""Keep the learned unused-finger posture through normal thumb/index release.

Only the phase features for unused digits stop advancing late in the episode.
Their measured joint feedback remains live; active fingers/wrists use real time.
"""
import numpy as np
from posture_feedback_policy_v3 import PosturePolicy as AnatomicalPolicy
from posture_feedback_policy_v4 import PosturePolicy as GraspPolicy


class PosturePolicy:
    def __init__(self,checkpoint):
        with np.load(checkpoint,allow_pickle=False) as z:has_grasp='grasp_wrist_actuator_ids' in z
        self.parent=(GraspPolicy if has_grasp else AnatomicalPolicy)(checkpoint)
        self.data=self.parent.data;self.qadr=self.parent.qadr;self.vadr=self.parent.vadr;self.duration=self.parent.duration

    def __call__(self,next_time_s,qpos,qvel):
        output=self.parent(next_time_s,qpos,qvel)
        freeze=float(self.data.get('posture_hold_phase_s',14.2))
        if next_time_s>freeze:
            held=self.parent(freeze,qpos,qvel)
            ids=self.data['posture_actuator_ids'];output[ids]=held[ids]
        return output
