"""Carry-phase wrist correction wrapped around an existing component actor (prototype, 2026-10-20).

The base learned actor is unchanged. During a fitted time window the commanded pose of the hand
that carries the pot is rotated by a constant world-frame correction about a pivot fixed in the hand
frame, and translated by a constant world offset, both ramped in/out smoothly. The parameters are
fitted offline from one previous physical rollout (mean pot tilt / sag relative to the reference).
Runtime inputs are the same as the base actor: time, measured joint positions and velocities.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from component_policy import ComponentPolicy
from release_feedback_policy import smooth
from wrist_reference import continuous_euler_xyz


class CarryTiltPolicy:
    def __init__(self, checkpoint, params):
        self.base = ComponentPolicy(checkpoint)
        self.params = dict(json.load(open(params))) if not isinstance(params, dict) else dict(params)
        for name in ('data', 'duration', 'qadr', 'vadr', 'output_decimals'):
            setattr(self, name, getattr(self.base, name))
        p = self.params
        self.pos_ids = np.asarray(p['pos_actuator_ids'], int)
        self.rot_ids = np.asarray(p['rot_actuator_ids'], int)
        # Constant correction (round 1-3) or time tables (ILC rounds): tables are ungated world-frame
        # corrections sampled at table_time_s and linearly interpolated.
        if 'table_time_s' in p:
            self.tt = np.asarray(p['table_time_s'], float)
            self.rot_table = np.asarray(p['rotation_table'], float); self.pos_table = np.asarray(p['translation_table'], float)
        else:
            self.tt = None
            self.rotvec = np.asarray(p['rotation_vector'], float) * float(p['rot_gain'])
            self.dpos = np.asarray(p['translation_m'], float) * float(p['pos_gain'])
        self.finger_ids = np.asarray(p.get('finger_actuator_ids', []), int)
        self.finger_bias = np.asarray(p.get('finger_bias_rad', []), float)
        self.pivot = np.asarray(p['pivot_hand_frame_m'], float) * float(p['pivot_fraction'])
        self.t0, self.t1, self.ramp = float(p['start_s']), float(p['hold_end_s']), float(p['ramp_s'])
        self.architecture = self.base.architecture + ' + fitted carry wrist correction'

    def gate(self, t):
        return smooth((t - self.t0) / self.ramp) * (1. - smooth((t - self.t1) / self.ramp))

    def __call__(self, next_time_s, qpos, qvel):
        u = self.base(next_time_s, qpos, qvel)
        g = self.gate(next_time_s)
        if g > 0.:
            if self.tt is None:
                rotvec, dpos = self.rotvec, self.dpos
            else:
                rotvec = np.array([np.interp(next_time_s, self.tt, self.rot_table[:, k]) for k in range(3)])
                dpos = np.array([np.interp(next_time_s, self.tt, self.pos_table[:, k]) for k in range(3)])
            pos = u[self.pos_ids].copy(); ang = u[self.rot_ids].copy()
            R = Rotation.from_euler('XYZ', ang); Rc = Rotation.from_rotvec(g * rotvec)
            c = pos + R.apply(self.pivot)
            u[self.pos_ids] = c + Rc.apply(pos - c) + g * dpos
            if len(self.finger_ids):
                u[self.finger_ids] = u[self.finger_ids] + g * self.finger_bias
            u[self.rot_ids] = continuous_euler_xyz(Rotation.from_matrix((Rc * R).as_matrix()[None]), ang)[0]
            d = self.data
            if self.output_decimals is not None:
                u = np.round(u, decimals=self.output_decimals)
            u = np.clip(u, d['ctrl_lo'], d['ctrl_hi'])
        return u
