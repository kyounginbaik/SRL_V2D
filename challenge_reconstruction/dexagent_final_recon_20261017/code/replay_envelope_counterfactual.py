#!/usr/bin/env python3
"""Fixed-parameter left-envelope gap-fill diagnostic; no candidate search."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d, binary_closing

from refine_grasp_cem import Experiment, TR, execute_reference, evaluate_files, dump
from grasp_stability import audit_grasp_stability, intervals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-gap-s', type=float, default=1.5)
    a = p.parse_args()
    a.run, a.output = a.run.resolve(), a.output.resolve()
    if a.output.exists():
        raise FileExistsError(a.output)
    winner = json.loads((a.run/'result.json').read_text())
    settings = json.loads((a.run/'settings.json').read_text())
    exp = Experiment(settings['spec'])
    # Restore the recorded v2 envelope definition explicitly, even if future
    # experiments adopt a different default. The command identity check is final.
    row = pq.read_table(Path(exp.spec['source'])/'reference.parquet').to_pylist()[0]
    raw_by_side = {}
    ramp = np.minimum(np.arange(exp.T)/max(1., .3*exp.fps), 1.)
    for side in ('left', 'right'):
        normals = np.asarray(row[f'mano_{side}_object_contact_normals'], float)
        active = np.any(np.linalg.norm(normals, axis=-1) > .1, axis=1)
        raw_by_side[side] = active
        if active.sum() >= 3:
            closed = binary_closing(active, structure=np.ones(max(2, round(.3*exp.fps))))
            envelope = gaussian_filter1d(closed.astype(float), max(1., .12*exp.fps))
        else:
            envelope = np.ones(exp.T)
        exp.envelope[side] = np.clip(envelope*ramp, 0., 1.)
    x = np.array(winner['parameters'])
    with np.load(winner['reference'], allow_pickle=False) as z:
        before = z['qpos']
    identity_error = float(np.max(np.abs(exp.commands(x)-before)))
    if identity_error > 1e-12:
        raise ValueError(f'Cannot reproduce recorded command trajectory: {identity_error}')
    filled = raw_by_side['left'].copy()
    times = np.arange(exp.T)/exp.fps
    holes = []
    for start, stop in intervals(~filled, times):
        if start > 0 and stop < exp.T and (stop-start)/exp.fps <= a.max_gap_s:
            filled[start:stop] = True
            holes.append(dict(start_s=float(times[start]), end_s=float(times[stop])))
    closed = binary_closing(filled, structure=np.ones(max(2, round(.3*exp.fps))))
    exp.envelope['left'] = np.clip(gaussian_filter1d(closed.astype(float), max(1., .12*exp.fps))*ramp, 0., 1.)
    after = exp.commands(x)
    a.output.mkdir(parents=True)
    path = TR.save_reference(a.output/'reference.npz', exp.model, after, exp.ref['labels'], exp.fps,
                            exp.ref['segment'], exp.ref['segments'], source='Fixed-winner left envelope gap-fill diagnostic',
                            arms_coupled=False, floating_runtime=True, no_retiming=True)
    execution = execute_reference(exp.spec['scene'], path, a.output/'execution',
                                  expected_duration_s=(exp.T-1)/exp.fps)
    # run = B/refinement_20261017/tag/episode; B is three ancestors up.
    source = a.run.parents[2]/'episodes'/winner['episode']
    metrics = evaluate_files(exp.spec['source'], execution['trajectory'], source/'input_manifest.json',
                             execution['scene'], source/'metric_geometry.json')
    grasp = audit_grasp_stability(execution['scene'], execution['trajectory'])
    dump(a.output/'mp_sr.json', metrics)
    dump(a.output/'grasp_stability_substeps.json', grasp)
    dump(a.output/'result.json', dict(episode=winner['episode'], fixed_parameters=x, filled_left_gaps=holes,
        command_reproduction_error=identity_error, changed_command_max=float(np.max(np.abs(after-before))),
        execution=execution, metrics=metrics, counterfactual=True, learned_policy=False,
        interpretation='Same winning parameters and physics; only the bounded interior left contact-envelope gaps are filled.'))


if __name__ == '__main__':
    main()
