#!/usr/bin/env python3
"""Retention re-plan (one hand, smooth constant offset inside [start, stop]) with an extra terminal term: the target
object's mean position/orientation error over the final TAIL_S seconds (so a late tilt after set-down is punished, not
only diluted in the full-horizon mean). Same search driver as retention.py."""
from pathlib import Path
import argparse, json
import numpy as np
from physics_retention_search import RetentionExperiment
from refine_grasp_cem import servo_rollout, angular_errors
from pickup_optimizer import optimize


class UprightRetention:
    def __init__(self, spec):
        self.inner = RetentionExperiment(spec); self.m = self.inner.model
        for key in ['ref', 'fps', 'dim', 'bound']: setattr(self, key, getattr(self.inner, key))
        self.scale = self.inner.scale * spec['exploration_multiplier']
        self.target = self.inner.names.index(spec['target_object']); self.tail = int(round(spec['tail_s'] * self.inner.fps)); self.w = spec['tail_weight']

    def commands(self, x): return self.inner.commands(x), {}

    def evaluate(self, x, return_trace=False):
        e = self.inner; Q = e.commands(x)
        roll = servo_rollout(e.model, Q, e.fps, qpos_start=e.initial, record_substeps=False)
        if roll['unstable'] or len(roll['qpos']) != e.T: return dict(cost=1e9, unstable=True)
        p, q = e.object_poses(roll['qpos']); ep = np.linalg.norm(p - e.tp, axis=-1); er = angular_errors(q[:, None], e.tq[:, None], e.axes, e.kinds)[:, 0]
        pos, ang = ep.mean(0), er.mean(0); ratio = np.maximum(pos / .03, ang / np.deg2rad(30))
        cost = float(ratio.max() + .25 * ratio.mean()); hand_pen = float(np.max(roll['hand_pen_m'])); cost += 100 * max(0., hand_pen - .01)
        k = self.target; tail_pos = float(ep[-self.tail:, k].mean()); tail_ang = float(er[-self.tail:, k].mean())
        cost += self.w * (tail_pos / .03 + tail_ang / np.deg2rad(30))
        pr = e.protect; viol = max(pos[pr] / .03, ang[pr] / np.deg2rad(30)) - 1.; cost += 100 * max(0., viol)
        cost += .001 * float(np.mean((np.asarray(x) / e.bound) ** 2))
        return dict(cost=cost, mean_position_cm=(100 * pos).tolist(), mean_orientation_deg=np.rad2deg(ang).tolist(), tail_position_cm=100 * tail_pos,
                    tail_orientation_deg=float(np.rad2deg(tail_ang)), max_hand_pen_m=hand_pen, mp_sr=float(np.all((pos < .03) & (ang < np.deg2rad(30)))),
                    protected_object_pass=bool(viol < 0.), unstable=False)


def run(source, episode_dir, output, start_s, stop_s, side, target_object, protect_object, seed, scale, tail_s, tail_weight, samples, iterations, workers):
    prev = json.loads((source / 'result.json').read_text()); manifest = json.loads((episode_dir / 'input_manifest.json').read_text())
    spec = dict(scene=prev['execution']['scene'], reference=str((source / 'reference.npz').resolve()), original_reference=str((episode_dir / 'tracking_reference.npz').resolve()),
                source=manifest['source'], geometry=str((episode_dir / 'metric_geometry.json').resolve()), manifest=str((episode_dir / 'input_manifest.json').resolve()),
                start_s=start_s, stop_s=stop_s, side=side, protect_object=protect_object, target_object=target_object, ramp_s=.3, exploration_multiplier=scale,
                tail_s=tail_s, tail_weight=tail_weight, episode=manifest['episode'])
    optimize(UprightRetention, spec, output, samples=samples, iterations=iterations, workers=workers, seed=seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'episode-dir', 'output']: p.add_argument('--' + key, type=Path, required=True)
    for key in ['start-s', 'stop-s']: p.add_argument('--' + key, type=float, required=True)
    p.add_argument('--side', required=True); p.add_argument('--target-object', required=True); p.add_argument('--protect-object', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--scale', type=float, default=.5); p.add_argument('--tail-s', type=float, default=1.5)
    p.add_argument('--tail-weight', type=float, default=1.0); p.add_argument('--samples', type=int, default=96); p.add_argument('--iterations', type=int, default=12); p.add_argument('--workers', type=int, default=16)
    run(**vars(p.parse_args()))
