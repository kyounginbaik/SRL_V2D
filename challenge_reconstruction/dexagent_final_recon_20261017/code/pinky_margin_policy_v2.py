"""Learned actor with small fitted target margins for two pinky joints.

The margins are optimized by complete physical rollouts. They change robot
commands only; source joint bounds, servo forces, scene and initial state stay
unchanged. This wrapper retains every learned parameter of its base actor.
"""
import importlib
from pathlib import Path
import numpy as np


class PinkyMarginPolicy:
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
        gate = np.ones(len(d['margin_actuator_ids']))
        # Startup correction fades smoothly before the first pickup.
        early = d.get('margin_startup_only', np.zeros(len(gate), dtype=bool))
        x = np.clip((float(d.get('margin_startup_end_s', .8))-next_time_s)/.4, 0., 1.)
        gate[early] = x*x*(3.-2.*x)
        output[d['margin_actuator_ids']] += gate*d['margin_bias_rad']
        return np.clip(output,d['ctrl_lo'],d['ctrl_hi'])
