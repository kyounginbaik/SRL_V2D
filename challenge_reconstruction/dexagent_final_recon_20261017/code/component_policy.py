"""One explicit actor registry for the incremental submission components.

Only checkpoint fields select the already trained architecture. No source
reference or future object state is read during inference.
"""
import importlib
import numpy as np


ACTORS = {
    'learned_feedback_policy': 'LearnedPolicy',
    'posture_feedback_policy': 'PosturePolicy',
    'posture_feedback_policy_v3': 'PosturePolicy',
    'posture_feedback_policy_v4': 'PosturePolicy',
    'joint_margin_policy': 'JointMarginPolicy',
    'pinky_margin_policy_v2': 'PinkyMarginPolicy',
}


class ComponentPolicy:
    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as z:
            keys = set(z.files)
            if 'margin_actuator_ids' in keys:
                base = str(z['margin_base_module'])
                if base not in ACTORS or str(z['margin_base_class']) != ACTORS[base]:
                    raise ValueError('Unsupported trained margin base architecture')
                module = ('pinky_margin_policy_v2' if 'margin_startup_only' in keys
                          else 'joint_margin_policy')
            elif 'grasp_wrist_actuator_ids' in keys:
                module = 'posture_feedback_policy_v4'
            elif 'posture_joint_bias_rad' in keys:
                module = 'posture_feedback_policy_v3'
            elif 'posture_actuator_ids' in keys:
                module = 'posture_feedback_policy'
            else:
                module = 'learned_feedback_policy'
        self.actor = getattr(importlib.import_module(module), ACTORS[module])(checkpoint)
        self.architecture = module + '.' + ACTORS[module]
        for name in ('data', 'duration', 'qadr', 'vadr'):
            setattr(self, name, getattr(self.actor, name))
        self.output_decimals = self.data.get('component_output_decimals')
        if self.output_decimals is not None:
            self.output_decimals = int(self.output_decimals)
            if not 6 <= self.output_decimals <= 12:
                raise ValueError('Command precision must preserve micrometer-scale motion or better')
            self.architecture += f' + decimal command precision {self.output_decimals}'

    def __call__(self, next_time_s, qpos, qvel):
        output = self.actor(next_time_s, qpos, qvel)
        if self.output_decimals is not None:
            output = np.round(output, decimals=self.output_decimals)
            output = np.clip(output, self.data['ctrl_lo'], self.data['ctrl_hi'])
        return output
