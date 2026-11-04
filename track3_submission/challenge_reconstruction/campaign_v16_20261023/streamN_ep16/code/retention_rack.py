#!/usr/bin/env python3
"""Stream N: rackscan_20261021/code/retention.py (same constant-offset right-hand retention CEM, rack protected) with an extra
rack term in the cost: cost += RACK_W * (rack mean position error cm / 3 + rack mean orientation error deg / 30), so the search trades
spoon error against rack displacement instead of treating the rack only as a pass/fail constraint.
usage: same args as retention.py plus --rack-weight W"""
from pathlib import Path
import argparse, json
from physics_retention_search import RetentionExperiment
from pickup_optimizer import optimize


class RackRetention:
    W = 1.0
    def __init__(self, spec):
        self.inner = RetentionExperiment(spec); self.m = self.inner.model; self.W = float(spec.get('rack_weight', 1.0))
        for key in ['ref', 'fps', 'dim', 'bound']: setattr(self, key, getattr(self.inner, key))
        self.scale = self.inner.scale * spec['exploration_multiplier']; self.rack = self.inner.names.index('plastic_dish_rack')
    def commands(self, x): return self.inner.commands(x), {}
    def evaluate(self, x, return_trace=False):
        r = self.inner.evaluate(x)
        if r.get('unstable'): return r
        r['cost_before_rack_term'] = r['cost']
        r['cost'] += self.W * (r['mean_position_cm'][self.rack] / 3. + r['mean_orientation_deg'][self.rack] / 30.)
        return r


def run(source, episode_dir, output, start_s, stop_s, protect_object, seed, scale, samples, iterations, workers, rack_weight):
    prev = json.loads((source / 'result.json').read_text()); manifest = json.loads((episode_dir / 'input_manifest.json').read_text())
    spec = dict(scene=prev['execution']['scene'], reference=str((source / 'reference.npz').resolve()),
                original_reference=str((episode_dir / 'tracking_reference.npz').resolve()), source=manifest['source'],
                geometry=str((episode_dir / 'metric_geometry.json').resolve()), manifest=str((episode_dir / 'input_manifest.json').resolve()),
                start_s=start_s, stop_s=stop_s, side='right', protect_object=protect_object, ramp_s=.3, exploration_multiplier=scale,
                episode=manifest['episode'], rack_weight=rack_weight)
    optimize(RackRetention, spec, output, samples=samples, iterations=iterations, workers=workers, seed=seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'episode-dir', 'output']: p.add_argument('--' + key, type=Path, required=True)
    for key in ['start-s', 'stop-s']: p.add_argument('--' + key, type=float, required=True)
    p.add_argument('--protect-object', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--scale', type=float, default=.25); p.add_argument('--samples', type=int, default=96)
    p.add_argument('--iterations', type=int, default=8); p.add_argument('--workers', type=int, default=16); p.add_argument('--rack-weight', type=float, default=1.0)
    run(**{k.replace('-', '_'): v for k, v in vars(p.parse_args()).items()})
