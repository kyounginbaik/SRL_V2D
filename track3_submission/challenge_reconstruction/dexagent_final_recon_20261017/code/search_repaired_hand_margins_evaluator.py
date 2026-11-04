"""Margin evaluator for an explicitly versioned reconstruction scene.

Original trajectory targets stay immutable. The scene initial state is checked
against the learned checkpoint's frozen scene hash, not falsely equated with
its known-invalid reconstructed object pose. No physics state is overwritten.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from refine_grasp_cem import Experiment, TR, E
from evaluate_mp_sr import validate_geometry
from pinky_margin_policy_v2 import PinkyMarginPolicy
from search_pinky_margins_v2 import Search as OriginalSearch
from learned_feedback_policy import sha

class FrozenMetric:
    object_poses=Experiment.object_poses
    def __init__(self,scene,inp):
        self.model=mujoco.MjModel.from_xml_path(scene)
        ref=TR.load_reference(str(inp/'tracking_reference.npz'),self.model)
        self.base=ref['qpos'].copy();self.fps=ref['fps']
        self.objects=E.object_free_joints(self.model);self.names=list(self.objects)
        self.anchors,self.axes,self.kinds=validate_geometry(self.names,json.loads((inp/'metric_geometry.json').read_text()))
        self.tp,self.tq=self.object_poses(self.base)

class Search(OriginalSearch):
    def __init__(self,spec):
        self.spec=spec;inp=Path(spec['episode_dir'])
        self.metric=FrozenMetric(spec['scene'],inp)
        self.m=self.metric.model;self.policy=PinkyMarginPolicy(spec['augmented_checkpoint'])
        if str(self.policy.data['scene_sha256'])!=sha(spec['scene']):
            raise ValueError('Versioned reconstruction scene differs from learned checkpoint physics')
        self.dt=float(self.m.opt.timestep);self.steps=round(1/self.metric.fps/self.dt)
        if abs(self.steps*self.dt-1/self.metric.fps)>1e-10:raise ValueError('Non-integral original frame period')
        self.fj=np.array([j for j in range(self.m.njnt) if (self.m.joint(j).name or '').startswith('robot_') and any(f'_{f}_' in self.m.joint(j).name for f in ['thumb','index','middle','ring','pinky'])])
        self.fq=self.m.jnt_qposadr[self.fj];self.frange=self.m.jnt_range[self.fj]
        self.dim=len(self.policy.data['margin_actuator_ids'])

WORK=None
def init(spec):
    global WORK
    WORK=Search(spec)
def evaluate(x):return WORK.evaluate(x)
