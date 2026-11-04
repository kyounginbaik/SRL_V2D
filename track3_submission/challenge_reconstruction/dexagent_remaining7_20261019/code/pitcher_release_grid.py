#!/usr/bin/env python3
"""Test physical release commands after a successful HaWoR-derived grasp.

Only the right robot hand changes. Every object target, initial object state,
and the already validated left cup command sequence is preserved exactly.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, copy, json, os, sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE, LEGACY, ROOT, write
sys.path.insert(0, str(LEGACY/'code'))
sys.path.insert(0, str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H, tracking as TR
from physics_pickup_search_v2 import PickupExperiment, blend
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files


def trial(task):
    source, output, release, aperture, retreat = task
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    old = json.loads((Path(source)/'result.json').read_text())
    spec = copy.deepcopy(old['spec'])
    spec['times']['release'] = 100.
    exp = PickupExperiment(spec)
    Q, timing = exp.commands(np.asarray(old['parameters']))
    preserved = TR.load_reference(old['reference'], exp.m)['qpos']
    k = min(len(Q)-1, round(release*exp.fps))
    if aperture:
        d = mujoco.MjData(exp.m); d.qpos[:] = Q[k]; mujoco.mj_forward(exp.m, d)
        objr = Rotation.from_quat(Q[k, exp.oc+3:exp.oc+7], scalar_first=True)
        targets = {}
        for finger in H.FINGERS:
            p = d.site_xpos[exp.rig.fingertip_sites[finger]].copy()
            # Pitcher is asymmetric and its metric axis is intentionally zero.
            # Open horizontally away from the body, without assuming symmetry.
            radial = p-Q[k,exp.oc:exp.oc+3]
            radial[2] = 0.
            radial /= max(np.linalg.norm(radial), 1e-8)
            targets[finger] = p + aperture*radial
        H.finger_ik(exp.m, d, exp.rig, targets, iters=180)
        opened = d.qpos[exp.fcols].copy()
        Q[:,exp.fcols] += (opened-Q[:,exp.fcols])*blend((exp.times-release)/.4)[:,None]
        direction = Q[k,exp.wcols[:3]]-Q[k,exp.oc:exp.oc+3]
        direction[2] = 0.; direction /= max(np.linalg.norm(direction), 1e-8)
        Q[:,exp.wcols[:3]] += (retreat*direction+np.array([0,0,.03]))*blend((exp.times-release-.4)/.5)[:,None]
    assert np.isfinite(Q).all()
    other = np.setdiff1d(np.arange(exp.m.nq), exp.cols)
    assert np.array_equal(Q[:,other], preserved[:,other])
    assert np.array_equal(Q[exp.times<8.6], preserved[exp.times<8.6])
    Q[:,exp.cols] = np.clip(Q[:,exp.cols], exp.limits[:,0], exp.limits[:,1])
    path = TR.save_reference(out/'reference.npz', exp.m, Q, exp.ref['labels'], exp.fps,
        exp.ref['segment'], exp.ref['segments'], source='HaWoR-derived physical pitcher grip; safe release; unchanged left cup and object targets', floating_runtime=True)
    execution = execute_reference(spec['scene'], path, out/'execution')
    metrics = evaluate_files(spec['source'], execution['trajectory'], spec['manifest'], execution['scene'], spec['geometry'])
    report = dict(episode='episode_000020', reference=str(path), execution=execution, metrics=metrics,
        release_seconds=release, aperture_m=aperture, retreat_m=retreat, prefix_preserved_until=8.6,
        object_targets_unchanged=True, left_commands_unchanged=True, learned_policy=False, submission_ready=False)
    write(out/'result.json', report)
    print(json.dumps(dict(output=str(out), mp_sr=metrics['reference_proxy_mp_sr'], objects=metrics['official_style_reference_proxy']['objects'])), flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4); args=p.parse_args()
    tasks=[]
    for name, release, aperture, retreat in [('hold',10.1,0.,0.),('release86',8.6,.015,.08),('release90',9.,.015,.08),('release92',9.2,.022,.10)]:
        tasks.append((str(args.source), str(args.output/name), release, aperture, retreat))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(trial, tasks))
    write(args.output/'summary.json', [dict(output=r['reference'], metrics=r['metrics']) for r in results])
