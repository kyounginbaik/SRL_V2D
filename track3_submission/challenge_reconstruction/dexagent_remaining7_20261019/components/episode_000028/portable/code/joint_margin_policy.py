"""Learned actor with small fitted target margins for two pinky joints.

The margins are optimized by complete physical rollouts. They change robot
commands only; source joint bounds, servo forces, scene and initial state stay
unchanged. This wrapper retains every learned parameter of its base actor.
"""
import importlib
from pathlib import Path
import numpy as np


class JointMarginPolicy:
    def __init__(self,checkpoint):
        with np.load(checkpoint,allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        module=importlib.import_module(str(d['margin_base_module']))
        self.base=getattr(module,str(d['margin_base_class']))(checkpoint)
        self.data=self.base.data
        self.duration=self.base.duration;self.qadr=self.base.qadr;self.vadr=self.base.vadr

    def __call__(self,next_time_s,qpos,qvel):
        output=self.base(next_time_s,qpos,qvel)
        d=self.data
        output[d['margin_actuator_ids']]+=d['margin_bias_rad']
        return np.clip(output,d['ctrl_lo'],d['ctrl_hi'])
