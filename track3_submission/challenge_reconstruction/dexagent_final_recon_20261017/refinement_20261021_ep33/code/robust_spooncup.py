#!/usr/bin/env python3
"""Robust spoon-in-cup retention: cost = WORST over the nominal rollout and K-1 rollouts whose right-hand finger commands carry
small fixed offsets inside the window (ep15 robust_retention idea) of the SpoonCupRetention cost (tracking + spoon-in-cup term).
The executed plan is the nominal one. Goal: a grip during the scoops that does not release the spoon on few-mrad finger errors."""
from pathlib import Path
import argparse, json
import numpy as np
from retention_spooncup import SpoonCupRetention
from pickup_optimizer import optimize


class RobustSpoonCup(SpoonCupRetention):
    def __init__(self, spec):
        super().__init__(spec); rng = np.random.default_rng(int(spec.get('robust_seed', 7))); nf = self.dim - 6
        self.offsets = [np.zeros(nf)] + [rng.normal(size=nf) * float(spec['finger_noise_rad']) for _ in range(int(spec['robust_k']) - 1)]
    def evaluate(self, x, return_trace=False):
        worst = None
        for off in self.offsets:
            y = np.asarray(x, float).copy(); y[6:] = y[6:] + off; r = super().evaluate(y)
            if worst is None or r['cost'] > worst['cost']: worst = r
        worst = dict(worst); worst['robust_k'] = len(self.offsets); return worst


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'episode-dir', 'output', 'spoon-mesh', 'cup-mesh']: p.add_argument('--' + key, type=Path, required=True)
    for key in ['start-s', 'stop-s']: p.add_argument('--' + key, type=float, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--scale', type=float, default=.5); p.add_argument('--cup-weight', type=float, default=1.)
    p.add_argument('--robust-k', type=int, default=3); p.add_argument('--finger-noise', type=float, default=.005)
    p.add_argument('--samples', type=int, default=96); p.add_argument('--iterations', type=int, default=10); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); prev = json.loads((a.source / 'result.json').read_text()); man = json.loads((a.episode_dir / 'input_manifest.json').read_text())
    spec = dict(scene=prev['execution']['scene'], reference=str((a.source / 'reference.npz').resolve()), original_reference=str((a.episode_dir / 'tracking_reference.npz').resolve()),
                source=man['source'], geometry=str((a.episode_dir / 'metric_geometry.json').resolve()), manifest=str((a.episode_dir / 'input_manifest.json').resolve()),
                start_s=a.start_s, stop_s=a.stop_s, side='right', protect_object='blue_cup', spoon='wooden_spoon', cup='blue_cup', ramp_s=.3, exploration_multiplier=a.scale,
                cup_weight=a.cup_weight, engage_m=.07, spoon_mesh=str(a.spoon_mesh), cup_mesh=str(a.cup_mesh), robust_k=a.robust_k, finger_noise_rad=a.finger_noise, episode=man['episode'])
    optimize(RobustSpoonCup, spec, a.output, samples=a.samples, iterations=a.iterations, workers=a.workers, seed=a.seed)
