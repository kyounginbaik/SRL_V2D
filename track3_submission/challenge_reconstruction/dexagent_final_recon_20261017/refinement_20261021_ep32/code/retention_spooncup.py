#!/usr/bin/env python3
"""Retention re-plan (holding hand, smooth constant offset inside [start, stop]) with an extra spoon-in-cup term: during the
frames where the reference spoon bowl is engaged with the cup (bowl point within ENGAGE_M of the cup rim centre), the bowl
point expressed in the ACHIEVED cup frame must match the reference bowl point expressed in the reference cup frame.
Same CEM driver as retention.py / retention_upright.py; outcomes are scored on actual rollouts (robust to contact chaos)."""
from pathlib import Path
import argparse, json
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
from physics_retention_search import RetentionExperiment
from refine_grasp_cem import servo_rollout, angular_errors
from pickup_optimizer import optimize


def spoon_geometry(spoon_mesh, cup_mesh):
    sv = np.asarray(trimesh.load(spoon_mesh, force='mesh').vertices); s = sv - sv.mean(0); w, U = np.linalg.eigh(s.T @ s); t = s @ U[:, -1]
    thick = lambda m: np.ptp(s[m] @ U[:, 1])
    hi, lo = t > np.percentile(t, 85), t < np.percentile(t, 15); bowl = sv[hi] if thick(hi) > thick(lo) else sv[lo]
    cv = np.asarray(trimesh.load(cup_mesh, force='mesh').vertices); c = cv - cv.mean(0); w, U = np.linalg.eigh(c.T @ c)
    g = [abs(w[k]-w[(k+1)%3]) + abs(w[k]-w[(k+2)%3]) - abs(w[(k+1)%3]-w[(k+2)%3]) for k in range(3)]; ax = U[:, int(np.argmax(g))]; h = c @ ax
    rad = np.linalg.norm(c - np.outer(h, ax), axis=1)
    if rad[h > np.percentile(h, 92)].min() < rad[h < np.percentile(h, 8)].min(): ax, h = -ax, -h
    rim = cv.mean(0) + ax * h.max()
    return bowl.mean(0), rim


class SpoonCupRetention:
    def __init__(self, spec):
        self.inner = e = RetentionExperiment(spec); self.m = e.model
        for key in ['ref', 'fps', 'dim', 'bound']: setattr(self, key, getattr(e, key))
        self.scale = e.scale * spec['exploration_multiplier']; self.w = spec['cup_weight']
        if spec.get('finger_exploration') is not None: self.scale[6:] = e.scale[6:] * float(spec['finger_exploration'])   # wrist and finger exploration decoupled
        self.si, self.ci = e.names.index(spec['spoon']), e.names.index(spec['cup'])
        self.tip, self.rim = spoon_geometry(spec['spoon_mesh'], spec['cup_mesh'])
        self.anch = np.asarray(e.anchors)
        bt, bc = self.body(e.tp, e.tq)
        self.ref_local = self.local(bt, bc)
        rim_w = np.einsum('tij,j->ti', bc[1], self.rim) + bc[0]
        self.engaged = np.linalg.norm(bt - rim_w, axis=1) < spec['engage_m']

    def body(self, p, q):
        Rm = lambda j: Rot.from_quat(q[:, j][:, [1, 2, 3, 0]]).as_matrix()
        Rs, Rc = Rm(self.si), Rm(self.ci)
        os_ = p[:, self.si] - np.einsum('tij,j->ti', Rs, self.anch[self.si]); oc = p[:, self.ci] - np.einsum('tij,j->ti', Rc, self.anch[self.ci])
        tip_w = np.einsum('tij,j->ti', Rs, self.tip) + os_
        return tip_w, (oc, Rc)

    def local(self, tip_w, cup):
        oc, Rc = cup; return np.einsum('tji,tj->ti', Rc, tip_w - oc)

    def commands(self, x): return self.inner.commands(x), {}

    def evaluate(self, x, return_trace=False):
        e = self.inner; Q = e.commands(x)
        roll = servo_rollout(e.model, Q, e.fps, qpos_start=e.initial, record_substeps=False)
        if roll['unstable'] or len(roll['qpos']) != e.T: return dict(cost=1e9, unstable=True)
        p, q = e.object_poses(roll['qpos']); ep = np.linalg.norm(p - e.tp, axis=-1); er = angular_errors(q[:, None], e.tq[:, None], e.axes, e.kinds)[:, 0]
        pos, ang = ep.mean(0), er.mean(0); ratio = np.maximum(pos / .03, ang / np.deg2rad(30))
        cost = float(ratio.max() + .25 * ratio.mean()); hand_pen = float(np.max(roll['hand_pen_m'])); cost += 100 * max(0., hand_pen - .01)
        tw, cup = self.body(p, q); loc = self.local(tw, cup); d = np.linalg.norm(loc - self.ref_local, axis=1)
        cup_err = float(d[self.engaged].mean()) if self.engaged.any() else 0.; cost += self.w * cup_err / .03
        pr = e.protect; viol = max(pos[pr] / .03, ang[pr] / np.deg2rad(30)) - 1.; cost += 100 * max(0., viol)
        cost += .001 * float(np.mean((np.asarray(x) / e.bound) ** 2))
        return dict(cost=cost, mean_position_cm=(100 * pos).tolist(), mean_orientation_deg=np.rad2deg(ang).tolist(), spoon_in_cup_error_cm=100 * cup_err,
                    engaged_frames=int(self.engaged.sum()), max_hand_pen_m=hand_pen, mp_sr=float(np.all((pos < .03) & (ang < np.deg2rad(30)))),
                    protected_object_pass=bool(viol < 0.), unstable=False)


def run(source, episode_dir, output, start_s, stop_s, side, protect_object, seed, scale, cup_weight, engage_m, spoon_mesh, cup_mesh, samples, iterations, workers, finger_scale=None):
    prev = json.loads((source / 'result.json').read_text()); manifest = json.loads((episode_dir / 'input_manifest.json').read_text())
    spec = dict(scene=prev['execution']['scene'], reference=str((source / 'reference.npz').resolve()), original_reference=str((episode_dir / 'tracking_reference.npz').resolve()),
                source=manifest['source'], geometry=str((episode_dir / 'metric_geometry.json').resolve()), manifest=str((episode_dir / 'input_manifest.json').resolve()),
                start_s=start_s, stop_s=stop_s, side=side, protect_object=protect_object, spoon='wooden_spoon', cup='blue_cup', ramp_s=.3, exploration_multiplier=scale,
                cup_weight=cup_weight, engage_m=engage_m, spoon_mesh=str(spoon_mesh), cup_mesh=str(cup_mesh), finger_exploration=finger_scale, episode=manifest['episode'])
    optimize(SpoonCupRetention, spec, output, samples=samples, iterations=iterations, workers=workers, seed=seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'episode-dir', 'output', 'spoon-mesh', 'cup-mesh']: p.add_argument('--' + key, type=Path, required=True)
    for key in ['start-s', 'stop-s']: p.add_argument('--' + key, type=float, required=True)
    p.add_argument('--side', default='right'); p.add_argument('--protect-object', default='blue_cup'); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--scale', type=float, default=.5); p.add_argument('--cup-weight', type=float, default=1.0); p.add_argument('--engage-m', type=float, default=.07)
    p.add_argument('--samples', type=int, default=96); p.add_argument('--iterations', type=int, default=12); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--finger-scale', type=float, default=None)
    run(**vars(p.parse_args()))
