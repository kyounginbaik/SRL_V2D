#!/usr/bin/env python3
"""Audit a completed CEM rollout and render its fixed-time comparison to the pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
_LIB = B/'.software_gl/usr/lib64'
if __name__ == '__main__':
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    if _LIB.is_dir() and str(_LIB) not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        env = dict(os.environ, LD_LIBRARY_PATH=str(_LIB)+':'+os.environ.get('LD_LIBRARY_PATH', ''))
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

import cv2
import numpy as np

from evaluate_mp_sr import evaluate_files
from grasp_stability import audit_grasp_stability
from check_cem_run import check_run

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--tag', default='cem_v1')
    p.add_argument('--run', type=Path, help='Explicit completed recording directory (result.json or report.json)')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()
    if args.episode not in (1, 3, 5, 8, 10) or '/' in args.tag or args.tag in ('.', '..'):
        raise ValueError('Expected one requested episode and a simple experiment tag')
    name = f'episode_{args.episode:06d}'
    source = B/'episodes'/name
    run = args.run.resolve() if args.run else B/'refinement_20261017'/args.tag/name
    if not any(run.is_relative_to(B/tag) for tag in ('refinement_20261017', 'refinement_20261018')):
        raise ValueError('Audit outputs must remain within this experiment workspace')
    report_path = run/'result.json' if (run/'result.json').exists() else run/'report.json'
    result = json.loads(report_path.read_text())
    if result.get('episode', name) != name:
        raise ValueError('Recording episode does not match requested reference')
    execution = result['execution']
    manifest = json.loads((source/'input_manifest.json').read_text())
    consistency = None
    if (run/'settings.json').exists() and (run/'result.json').exists():
        consistency = check_run(run)
        write(run/'audit/search_replay_consistency.json', consistency)
    else:
        write(run/'audit/recording_validation.json', dict(
            physics_verified=execution.get('physics_verified'),
            full_horizon=execution.get('execution_completed'),
            independent_replay=execution.get('independent_control_replay'),
            search_replay_comparison=None,
            note='Recorded execution audit; this trial is not a CEM optimizer result.'))
    metrics = evaluate_files(manifest['source'], execution['trajectory'], source/'input_manifest.json',
                             execution['scene'], source/'metric_geometry.json')
    write(run/'audit/mp_sr_independent.json', metrics)
    write(run/'audit/grasp_stability_frames.json', audit_grasp_stability(
        execution['scene'], execution['trajectory'], frames_only=True))
    videos = run/'videos'
    render_args = dict(episode_dir=str(source),
        baseline_report=str(B/'runs/pilot_v2'/name/'seed_44/baseline/execution_report.json'),
        refined_report=str(Path(execution['trajectory']).parent/'execution_report.json'), out_dir=str(videos),
        width=args.width, height=args.height, stride=args.stride, include_source=True)
    write(videos/'render_args.json', render_args)
    # The renderer launches itself with its software OpenGL library environment.
    subprocess.run([sys.executable, str(B/'code/render_rollouts.py'), '--args',
                    str(videos/'render_args.json')], check=True,
                   env=dict(os.environ, MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa'))
    capture = cv2.VideoCapture(str(videos/'reference_baseline_refined.mp4'))
    if not capture.isOpened():
        raise RuntimeError('Rendered comparison cannot be opened for frame inspection')
    preview = videos/'review_frames'; preview.mkdir(exist_ok=True)
    preview_images = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = float(execution['duration_s'])
    requested = [t for t in (1., 1.5, 2., 3., 5.) if t <= duration]
    full_review = np.arange(0., duration+1e-9, .5).tolist() + [duration]
    for t in sorted(set(requested+full_review)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(t*fps))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Missing rendered review frame at {t}s')
        cv2.imwrite(str(preview/f'time_{round(t*1000):06d}ms.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        if t in requested:
            preview_images.append(frame)
    capture.release()
    if preview_images:
        cv2.imwrite(str(videos/'contactsheet_requested.jpg'), np.vstack(preview_images),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
    write(run/'audit/completed.json', dict(episode=name, full_horizon=execution['execution_completed'],
        physics_verified=execution['physics_verified'], reference_proxy_mp_sr=metrics['reference_proxy_mp_sr'],
        search_replay_consistent=consistency['consistent'] if consistency else None,
        video=str(videos/'reference_baseline_refined.mp4'),
        note='Recorded physics comparison and reference-proxy score; no official held-out result or learned-policy certification'))


if __name__ == '__main__':
    main()
