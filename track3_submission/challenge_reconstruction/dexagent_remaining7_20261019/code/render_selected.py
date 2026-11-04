#!/usr/bin/env python3
"""Render measured new-episode states with their HaWoR reference and real RGB."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
LEGACY = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--refined', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--episode-dir', type=Path,
                   help='Prepared episode for this exact scene variant')
    args = p.parse_args()
    if (args.output / 'reference_baseline_refined.mp4').exists():
        raise FileExistsError('Preserve existing rendered video')
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.baseline / 'result.json').read_text())
    refined = json.loads((args.refined / 'result.json').read_text())
    for report in (baseline, refined):
        if not report['execution']['physics_verified']:
            raise ValueError('Only verified full physical recordings may be rendered')
    episode_dir = args.episode_dir or BASE / 'physics_collision_v2' / 'episodes' / f'episode_{args.episode:06d}'
    if not (episode_dir / 'preparation_complete.json').exists():
        raise FileNotFoundError('Prepared episode is missing: ' + str(episode_dir))
    payload = dict(episode_dir=str(episode_dir.resolve()),
        baseline_report=baseline['execution'], refined_report=refined['execution'],
        out_dir=str(args.output.resolve()), width=512, height=384,
        stride=args.stride, include_source=True)
    task = args.output / 'render_args.json'
    task.write_text(json.dumps(payload, indent=2) + '\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LP_NUM_THREADS='2',
        MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa',
        LD_LIBRARY_PATH=str(LEGACY / '.software_gl/usr/lib64') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    subprocess.run([sys.executable, str(LEGACY / 'code/render_rollouts.py'), '--args', str(task)],
        env=env, check=True)


if __name__ == '__main__':
    main()
