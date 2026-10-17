"""Independent finite release primitives for the two hands of a learned actor.

Each primitive holds only its own hand's learned phase after its fitted anchor;
proprioceptive feedback stays active. Runtime inputs contain no object poses,
reference trajectory, future state, or recorded action table. Existing single
release actors and archived checkpoints are unaffected by this new class.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pinky_margin_policy_v2 import PinkyMarginPolicy
from release_feedback_policy import smooth
from wrist_reference import continuous_euler_xyz


class MultiReleasePolicy:
    def __init__(self,checkpoint):
        self.base=PinkyMarginPolicy(checkpoint);self.data=self.base.data
        self.duration=self.base.duration;self.qadr=self.base.qadr;self.vadr=self.base.vadr
        self.architecture='multi_release_feedback_policy.MultiReleasePolicy'
        if str(self.data['multi_release_base_module'])!='pinky_margin_policy_v2':raise ValueError('Unsupported base actor')
        self.prefixes=self.data['multi_release_prefixes'].tolist();used=set()
        for p in self.prefixes:
            ids=set(self.data[p+'actuator_ids'].tolist())
            if used&ids:raise ValueError('Release primitives must control disjoint hands')
            used|=ids

    def __call__(self,next_time_s,qpos,qvel):
        d=self.data;result=self.base(next_time_s,qpos,qvel)
        for p in self.prefixes:
            start=float(d[p+'start_s'])
            if next_time_s<=start:continue
            ids=d[p+'actuator_ids'];held=self.base(start,qpos,qvel);result[ids]=held[ids]
            place=float(d.get(p+'place_duration_s',0.))
            if place:
                placement=smooth((next_time_s-start)/place)
                result[d[p+'translation_actuator_ids']]+=placement*d[p+'place_delta_m']
                if p+'place_rotation_vector' in d and np.any(d[p+'place_rotation_vector']):
                    ri=d[p+'rotation_actuator_ids'];angles=result[ri].copy()
                    desired=Rotation.from_euler('XYZ',angles)*Rotation.from_rotvec(placement*d[p+'place_rotation_vector'])
                    result[ri]=continuous_euler_xyz(Rotation.from_matrix(desired.as_matrix()[None]),angles)[0]
            opening=smooth((next_time_s-start-place)/float(d[p+'open_duration_s']))
            if p+'closed_bias_rad' in d:
                closing=smooth((next_time_s-start)/max(place,.12))
                result[d[p+'finger_actuator_ids']]+=(1-opening)*closing*d[p+'closed_bias_rad']
            result[d[p+'finger_actuator_ids']]+=opening*d[p+'open_delta_rad']
            retreat=smooth((next_time_s-start-place-float(d[p+'open_duration_s']))/float(d[p+'retreat_duration_s']))
            result[d[p+'translation_actuator_ids']]+=retreat*d[p+'retreat_m']
        result=np.clip(result,d['ctrl_lo'],d['ctrl_hi'])
        if 'component_output_decimals' in d:result=np.clip(np.round(result,int(d['component_output_decimals'])),d['ctrl_lo'],d['ctrl_hi'])
        return result
