"""Phase/proprioception actor with a fitted, supported release primitive.

Only finite robot-control parameters are stored. The right-hand learned phase is
held during release, while its measured-joint feedback remains active. There is
no reference trajectory, object observation or recorded-action table at runtime.
"""
import importlib
import numpy as np
from scipy.spatial.transform import Rotation
from wrist_reference import continuous_euler_xyz


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


class ReleasePolicy:
    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        module = importlib.import_module(str(saved['release_base_module']))
        self.base = getattr(module, str(saved['release_base_class']))(checkpoint)
        self.data = self.base.data
        self.duration = self.base.duration
        self.qadr, self.vadr = self.base.qadr, self.base.vadr
        self.architecture = 'release_feedback_policy.ReleasePolicy'

    def __call__(self, next_time_s, qpos, qvel):
        d = self.data
        result = self.base(next_time_s, qpos, qvel)
        start = float(d['release_start_s'])
        if next_time_s > start:
            ids = d['release_actuator_ids']
            held = self.base(start, qpos, qvel)
            result[ids] = held[ids]
            place_duration = float(d.get('release_place_duration_s', 0.))
            if place_duration:
                placement = smooth((next_time_s-start)/place_duration)
                result[d['release_translation_actuator_ids']] += placement*d['release_place_delta_m']
                if 'release_place_rotation_vector' in d and np.any(d['release_place_rotation_vector']):
                    rotation_ids=d['release_rotation_actuator_ids'];angles=result[rotation_ids].copy()
                    desired=Rotation.from_euler('XYZ',angles)*Rotation.from_rotvec(placement*d['release_place_rotation_vector'])
                    result[rotation_ids]=continuous_euler_xyz(Rotation.from_matrix(desired.as_matrix()[None]),angles)[0]
            opening = smooth((next_time_s-start-place_duration)/float(d['release_open_duration_s']))
            if 'release_closed_bias_rad' in d:
                closing = smooth((next_time_s-start)/max(place_duration,.12))
                result[d['release_finger_actuator_ids']] += (1-opening)*closing*d['release_closed_bias_rad']
            result[d['release_finger_actuator_ids']] += opening*d['release_open_delta_rad']
            retreat = smooth((next_time_s-start-place_duration-float(d['release_open_duration_s']))/float(d['release_retreat_duration_s']))
            result[d['release_translation_actuator_ids']] += retreat*d['release_retreat_m']
        result = np.clip(result, d['ctrl_lo'], d['ctrl_hi'])
        if 'component_output_decimals' in d:
            result = np.clip(np.round(result, int(d['component_output_decimals'])), d['ctrl_lo'], d['ctrl_hi'])
        return result
