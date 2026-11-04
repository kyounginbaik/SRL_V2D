"""Distil a physical teacher recording (result.json with execution.scene/trajectory) into registered LearnedPolicy actors over a
hyper-parameter grid and evaluate each with the UNCHANGED frozen evaluator (evaluate_policy_component.evaluate) in the teacher's scene
against the given episode dir. Writes <output>/actor_c{C}_p{G}_v{V}/{training,evaluation} and attempts.json / completed.json.
usage: actor_grid_sB.py --episode E --teacher DIR --episode-dir DIR --output DIR [--workers 4] [--decimals 6] [--grid "1024,0.01,0.0005;..."]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
DEFAULT_GRID = [(1024, .01, .0005), (512, .01, .0005), (1024, .03, .0005), (512, .03, .0005), (1024, .003, .0005), (2048, .01, .0005), (1024, .01, .0002), (512, .003, .0001)]

def write(p, r): p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(r, indent=2) + '\n')

def attempt(params):
    episode, teacher, epdir, output, centers, gain, velocity, decimals = params
    from learned_feedback_policy import train, sha
    from evaluate_policy_component import evaluate
    teacher, output = Path(teacher), Path(output); r = json.loads((teacher / 'result.json').read_text())
    scene, traj = r['execution']['scene'], r['execution']['trajectory']
    dest = output / f'actor_c{centers}_p{gain:g}_v{velocity:g}'; ck = dest / 'training/policy.npz'
    try:
        if not ck.exists():
            rep = train(scene, traj, dest / 'training', centers=centers, feedback_gain=gain, velocity_gain=velocity, seed=43)
            if decimals is not None:
                d = dict(np.load(ck, allow_pickle=False)); d['component_output_decimals'] = np.array(decimals); np.savez_compressed(ck, **d); rep['component_output_decimals'] = decimals
            rep.update(teacher=str(teacher), checkpoint_sha256=sha(ck)); json.dump(rep, open(dest / 'training/training_report.json', 'w'), indent=2)
        if not (dest / 'evaluation/result.json').exists(): evaluate(episode, scene, str(ck), str(dest / 'evaluation'), str(epdir))
        ev = json.loads((dest / 'evaluation/result.json').read_text()); obs = ev['metrics']['official_style_reference_proxy']['objects']
        return dict(episode=f'episode_{episode:06d}', teacher=str(teacher), checkpoint=str(ck), evaluation=str(dest / 'evaluation'), episode_dir=str(epdir),
                    passed=ev['metrics']['reference_proxy_mp_sr'] == 1. and ev['execution']['physics_verified'],
                    object_errors={k: dict(cm=v['mean_position_error_cm'][0], deg=v['mean_orientation_error_degrees'][0]) for k, v in obs.items()},
                    maximum_ratio=max(max(v['mean_position_error_cm'][0] / 3, v['mean_orientation_error_degrees'][0] / 30) for v in obs.values()), physical_replay_verified=ev['execution']['physics_verified'])
    except Exception as e:
        return dict(episode=f'episode_{episode:06d}', checkpoint=str(ck), passed=False, error=repr(e)[:300], maximum_ratio=99.)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--episode', type=int, required=True); p.add_argument('--teacher', required=True); p.add_argument('--episode-dir', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--decimals', type=int, default=6); p.add_argument('--grid'); a = p.parse_args()
    grid = [tuple(float(x) if i else int(x) for i, x in enumerate(g.split(','))) for g in a.grid.split(';')] if a.grid else DEFAULT_GRID
    args = [(a.episode, a.teacher, a.episode_dir, a.output, c, g, v, a.decimals) for c, g, v in grid]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        reports = []
        for row in pool.map(attempt, args):
            reports.append(row); write(Path(a.output) / 'attempts.json', reports); print(json.dumps(row), flush=True)
    passed = [r for r in reports if r['passed']]
    if passed:
        best = min(passed, key=lambda r: r['maximum_ratio']); write(Path(a.output) / 'completed.json', best); print('BEST', json.dumps(best))
    else: write(Path(a.output) / 'failed.json', dict(attempts=len(reports), all_failed=True))

if __name__ == '__main__': main()
