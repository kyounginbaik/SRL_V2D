#!/usr/bin/env python3
"""Robust small-step retention: the cost of a candidate is the WORST of the nominal rollout and K-1 rollouts whose right-hand
finger commands carry small fixed offsets (inside the window), so the selected plan does not depend on a knife-edge release.
The final executed plan is the nominal one (same as retention.py)."""
from pathlib import Path
import argparse, json
import numpy as np
from physics_retention_search import RetentionExperiment
from pickup_optimizer import optimize


class RobustRetention:
    def __init__(self, spec):
        self.inner = RetentionExperiment(spec); self.m = self.inner.model
        for key in ['ref', 'fps', 'dim', 'bound']: setattr(self, key, getattr(self.inner, key))
        self.scale = self.inner.scale * spec['exploration_multiplier']
        rng = np.random.default_rng(int(spec.get('robust_seed', 7))); nf = len(self.inner.cols[self.inner.side]) - 6
        self.offsets = [np.zeros(nf)] + [rng.normal(size=nf) * float(spec.get('finger_noise_rad', .005)) for _ in range(int(spec.get('robust_k', 3)) - 1)]
    def commands(self, x): return self.inner.commands(x), {}
    def evaluate(self, x, return_trace=False):
        worst = None
        for off in self.offsets:
            y = np.asarray(x, float).copy(); y[6:] = y[6:] + off      # finger offsets ride on the retention finger channels (windowed)
            r = self.inner.evaluate(y)
            if worst is None or r['cost'] > worst['cost']: worst = r
            if r.get('unstable'): break
        worst = dict(worst); worst['robust_k'] = len(self.offsets); return worst


def run(source, episode_dir, output, start_s, stop_s, protect_object, seed, scale=.25, samples=96, iterations=8, workers=16, robust_k=3, finger_noise=.005):
    prev = json.loads((source / 'result.json').read_text()); manifest = json.loads((episode_dir / 'input_manifest.json').read_text())
    spec = dict(scene=prev['execution']['scene'], reference=str((source / 'reference.npz').resolve()), original_reference=str((episode_dir / 'tracking_reference.npz').resolve()),
                source=manifest['source'], geometry=str((episode_dir / 'metric_geometry.json').resolve()), manifest=str((episode_dir / 'input_manifest.json').resolve()),
                start_s=start_s, stop_s=stop_s, side='right', protect_object=protect_object, ramp_s=.3, exploration_multiplier=scale, episode=manifest['episode'],
                robust_k=robust_k, finger_noise_rad=finger_noise)
    optimize(RobustRetention, spec, output, samples=samples, iterations=iterations, workers=workers, seed=seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'episode-dir', 'output']: p.add_argument('--' + key, type=Path, required=True)
    for key in ['start-s', 'stop-s']: p.add_argument('--' + key, type=float, required=True)
    p.add_argument('--protect-object', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--scale', type=float, default=.25); p.add_argument('--samples', type=int, default=96); p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--robust-k', type=int, default=3); p.add_argument('--finger-noise', type=float, default=.005)
    run(**vars(p.parse_args()))
