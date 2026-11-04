#!/usr/bin/env python3
"""Refine new HaWoR robot references and fit the established feedback actor.

Only physical robot commands are optimized. Full object targets, source timing,
scene home, geometry and actuator bounds remain fixed. Unsuccessful experiments
are retained as failures and are never promoted to submission components.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path('/data1/SRL_V2D')
BASE = ROOT / 'challenge_reconstruction/dexagent_remaining7_20261019'
LEGACY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017'
sys.path.insert(0, str(ROOT / 'third_party/DexAgent'))
sys.path.insert(0, str(LEGACY / 'code'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    if args.episode not in [14, 15, 16, 20, 25, 28, 30]:
        raise ValueError('This runner is scoped to the seven new episodes')
    import refine_grasp_cem as cem
    import temporal_refine as temporal
    from learned_feedback_policy import train
    from evaluate_policy_component import evaluate

    # Only the wrappers' input-directory variables change. Their physical
    # implementation and all previously prepared scenes remain untouched.
    cem.B = args.base
    temporal.B = args.base
    name = f'episode_{args.episode:06d}'
    episode_dir = args.base / 'episodes' / name
    output = args.output or args.base / 'refinement' / name
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'completed.json').exists():
        raise FileExistsError('Preserve completed refinement: ' + str(output))
    begin = time.monotonic()
    current = args.source_run.resolve()
    report = read(current / 'result.json')
    if not report['execution']['physics_verified']:
        raise ValueError('Input teacher has not passed physical replay')
    original = episode_dir / 'tracking_reference.npz'
    fallback = args.base / 'refinement_20261017/continuous_references' / (name + '.npz')
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if fallback.exists() and fallback.read_bytes() != original.read_bytes():
        raise ValueError('Previously staged fallback changed')
    if not fallback.exists():
        shutil.copy2(original, fallback)

    def reference_of(result):
        return result.get('reference') or result.get('search', {}).get('reference')

    if not reference_of(report):
        raise ValueError('Teacher command reference is missing')
    write(output / 'status.json', dict(stage='refinement', source_run=str(current),
        source_proxy_success=report['metrics']['reference_proxy_mp_sr'], hand_source='HaWoR'))
    if report['metrics']['reference_proxy_mp_sr'] != 1.:
        stage = output / 'full_hand_cem'
        if not (stage / 'result.json').exists():
            cem.optimize(args.episode, report['execution']['scene'], reference_of(report), stage,
                samples=args.samples, iterations=args.iterations, workers=args.workers, seed=args.seed)
        current, report = stage, read(stage / 'result.json')
    if report['metrics']['reference_proxy_mp_sr'] != 1.:
        stage = output / 'local_windows'
        if not (stage / 'result.json').exists():
            temporal.run(args.episode, current, stage, workers=args.workers,
                samples=args.samples, iterations=args.local_iterations, seed=args.seed + 1)
        current, report = stage, read(stage / 'result.json')
    write(output / 'selected_teacher.json', dict(directory=str(current), report=report))
    if report['metrics']['reference_proxy_mp_sr'] != 1.:
        write(output / 'status.json', dict(stage='needs_further_grasp_refinement',
            selected_teacher=str(current), metrics=report['metrics'],
            elapsed_seconds=time.monotonic() - begin, submission_ready=False))
        return False

    attempts = []
    # A successful command trajectory alone is not a policy. Fit real regression
    # weights, then execute the actor with measured proprioception from home.
    for centers, gain in [(512, .03), (1024, .03), (1024, .01), (2048, .03)]:
        attempt = output / f'actor_c{centers}_gain{gain:g}'
        checkpoint = attempt / 'training/policy.npz'
        if not checkpoint.exists():
            train(report['execution']['scene'], report['execution']['trajectory'], attempt / 'training',
                  centers=centers, feedback_gain=gain, seed=args.seed)
        if not (attempt / 'evaluation/result.json').exists():
            evaluate(args.episode, report['execution']['scene'], checkpoint,
                     attempt / 'evaluation', episode_dir)
        evaluated = read(attempt / 'evaluation/result.json')
        ok = bool(evaluated['execution']['physics_verified'] and
                  evaluated['metrics']['reference_proxy_mp_sr'] == 1.)
        attempts.append(dict(directory=str(attempt), passed=ok,
                             metrics=evaluated['metrics']))
        write(output / 'actor_attempts.json', attempts)
        if ok:
            done = dict(episode=name, hand_source='HaWoR', teacher=str(current),
                checkpoint=str(checkpoint), training_report=str(attempt / 'training/training_report.json'),
                evaluation=str(attempt / 'evaluation'), episode_dir=str(episode_dir),
                reference_proxy_mp_sr=1., physical_replay_verified=True,
                portable_validation_pending=True, submission_ready=False,
                elapsed_seconds=time.monotonic() - begin)
            write(output / 'completed.json', done)
            write(output / 'status.json', dict(stage='learned_actor_passed', **done))
            print(json.dumps(done), flush=True)
            return True
    write(output / 'status.json', dict(stage='teacher_passed_actor_needs_refinement',
        selected_teacher=str(current), attempts=attempts, submission_ready=False))
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--local-iterations', type=int, default=12)
    parser.add_argument('--seed', type=int, default=44)
    run(parser.parse_args())
