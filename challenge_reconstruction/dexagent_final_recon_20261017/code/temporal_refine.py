"""Refine bounded robot commands in short windows, scoring the full episode.

Every trial starts at the exact scene home. No prefix caching, object actuation,
retiming, mesh edits, or changed evaluation targets. A locally better grasp must
also improve the complete coupled two-object rollout before it is retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from refine_grasp_cem import B, Experiment, TR, dump, _freeze, execute_reference, evaluate_files

WORK = None


def init(spec):
    global WORK
    WORK = Experiment(spec)


def configure(exp, base, center, width):
    exp.base = base.copy()
    t = np.arange(exp.T) / exp.fps
    # Raised cosine has zero value and derivative at each edge.
    u = np.clip(np.abs(t - center) / (width / 2), 0., 1.)
    w = .5 + .5 * np.cos(np.pi * u)
    w[0] = 0.
    for side in ('left', 'right'):
        exp.rot[side] = Rotation.from_euler('XYZ', base[:, exp.cols[side][3:6]])
        exp.envelope[side] = w.copy()


def evaluate(task):
    base, center, width, x = task
    configure(WORK, base, center, width)
    report = WORK.evaluate(x)
    if not report['unstable']:
        # All candidates use the same physical objective, independent of how
        # many previously accepted local edits formed their current base.
        report['cost'] -= .001 * float(np.mean((x / WORK.bound)**2))
    return report


def run(episode, source_run, output, workers=16, samples=224, iterations=20, seed=49):
    name = f'episode_{episode:06d}'
    prev = Path(source_run).resolve(); out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'settings.json').exists():
        raise FileExistsError('Use a new output directory')
    inp = B/'episodes'/name
    manifest = json.loads((inp/'input_manifest.json').read_text())
    previous = json.loads((prev/'result.json').read_text())
    frozen_dir = out/'search_physics'; frozen_dir.mkdir()
    _, frozen = _freeze(previous['execution']['scene'], frozen_dir)
    spec = dict(scene=str(frozen), reference=str(prev/'reference.npz'),
                original_reference=str(inp/'tracking_reference.npz'),
                source=manifest['source'], geometry=str(inp/'metric_geometry.json'))
    exp = Experiment(spec); base = exp.base.copy()
    initial = exp.evaluate(np.zeros(56)); best_report = initial
    original_base = base.copy()
    protected = ((np.array(initial['mean_position_cm']) < 3.) &
                 (np.array(initial['mean_orientation_deg']) < 30.))
    rng = np.random.default_rng(seed); history = []; begin = time.monotonic()
    scale = np.tile([.004]*3 + [.05]*3 + [.09]*22, 2)
    centers = np.arange(1., (exp.T-1)/exp.fps, 1.5)
    widths = [2., 3.5, 5.]
    groups = [np.arange(0,6), np.arange(6,28), np.arange(28,34),
              np.arange(34,56), np.arange(28), np.arange(28,56), np.arange(56)]
    dump(out/'settings.json', dict(episode=name, spec=spec, samples=samples,
        iterations=iterations, workers=workers, seed=seed, learned_policy=False,
        algorithm='Local smooth window edits; coordinate and block proposals; exact full-episode physics',
        source_sha256={k:hashlib.sha256(Path(spec[k]).read_bytes()).hexdigest()
                       for k in ('scene','reference','original_reference','geometry')},
        optimizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    dump(out/'initial.json', initial)

    def admissible(r):
        return not r['unstable'] and bool(np.all(
            (np.array(r['mean_position_cm'])[protected] < 3.) &
            (np.array(r['mean_orientation_deg'])[protected] < 30.)))

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=init, initargs=(spec,)) as pool:
        for iteration in range(iterations):
            proposals = []
            # Pair signs using the same window and direction. Half test sparse
            # single-joint changes; half test coordinated finger/wrist changes.
            for pair in range(samples//2):
                center = float(centers[(pair + iteration) % len(centers)])
                width = float(widths[(pair//len(centers) + iteration) % len(widths)])
                x = np.zeros(56)
                if pair % 2 == 0:
                    j = (pair//2 + iteration*17) % 56
                    x[j] = scale[j]
                else:
                    g = groups[(pair//2 + iteration) % len(groups)]
                    x[g] = rng.normal(size=len(g)) * scale[g]
                proposals.extend([(base,center,width,x), (base,center,width,-x)])
            reports = list(pool.map(evaluate, proposals, chunksize=1))
            order = sorted(range(len(reports)), key=lambda j: reports[j]['cost'] if admissible(reports[j]) else 1e20)
            j = order[0]; improved = admissible(reports[j]) and reports[j]['cost'] < best_report['cost'] - 1e-8
            accepted = None
            if improved:
                _, center, width, x = proposals[j]
                configure(exp, base, center, width)
                base = exp.commands(x)
                best_report = reports[j]
                protected |= ((np.array(best_report['mean_position_cm']) < 3.) &
                              (np.array(best_report['mean_orientation_deg']) < 30.))
                accepted = dict(center_s=center, width_s=width, correction=x)
                TR.save_reference(out/'incumbent.npz', exp.model, base, exp.ref['labels'], exp.fps,
                    exp.ref['segment'], exp.ref['segments'], source=spec['reference'], floating_runtime=True, no_retiming=True)
            else:
                scale *= .75
            history.append(dict(iteration=iteration, best=best_report, improved=improved,
                accepted=accepted, elapsed_seconds=time.monotonic()-begin,
                candidate_successes=sum(r.get('mp_sr',0)==1 for r in reports),
                protected_rejections=sum(not admissible(r) for r in reports)))
            dump(out/'progress.json',dict(history=history)); print(json.dumps(history[-1], default=lambda x:x.tolist()),flush=True)
            if best_report['mp_sr'] == 1.:
                break
    configure(exp, base, 1., 2.)
    final = exp.evaluate(np.zeros(56))
    if not np.allclose(final['mean_position_cm'],best_report['mean_position_cm'],atol=1e-7,rtol=0):
        raise RuntimeError('Accepted reference does not reproduce the search rollout')
    path = TR.save_reference(out/'reference.npz',exp.model,base,exp.ref['labels'],exp.fps,
        exp.ref['segment'],exp.ref['segments'],source=f'Temporal physical refinement from {spec["reference"]}',
        floating_runtime=True,no_retiming=True)
    execution = execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(exp.T-1)/exp.fps)
    metrics = evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'mp_sr.json',metrics)
    dump(out/'result.json',dict(episode=name,initial=initial,best=final,history=history,
        reference=path,execution=execution,metrics=metrics,learned_policy=False,
        elapsed_seconds=time.monotonic()-begin,
        max_generalized_command_change=float(np.max(np.abs(base-original_base)))))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--episode',type=int,required=True);p.add_argument('--source-run',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--samples',type=int,default=224);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--seed',type=int,default=49)
    run(**vars(p.parse_args()))
