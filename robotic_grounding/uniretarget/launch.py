"""Launch an isolated seed on any Docker/NVIDIA Linux server (no host Python deps)."""
from pathlib import Path
import argparse
import hashlib
import os
import shlex
import socket
import subprocess

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, choices=(1, 3, 5, 6, 35), default=1)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--phase', choices=('full', 'smoke', 'pilot'), default='full')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--upper', choices=('gaussian', 'flashsac'), default='gaussian')
    p.add_argument('--image', default='uniretarget:chord100m')
    p.add_argument('--run-root', type=Path, default=None)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.seed < 0 or ',' in a.gpu:
        p.error('Use a nonnegative seed and exactly one GPU index or UUID')
    root = (a.run_root or REPO / '.uniretarget' / f'chord-consensus16-{a.upper}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'run', '--rm', '--init', '--user', f'{os.getuid()}:{os.getgid()}',
               '--shm-size', '2g', '-v', f'{REPO}:{REPO}', '-v', f'{root}:{root}',
               '-v', f'{PACKAGE.parent / "flash_chord"}:/workspace', '-w', '/workspace']
    if not a.prepare_only:
        command += ['--gpus', f'device={a.gpu}']
    variables = dict(HOME='/tmp', PYTHONUNBUFFERED='1', PYTHONPATH='/workspace/src',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4',
        NVIDIA_DRIVER_CAPABILITIES='all', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        PYGLET_HEADLESS='true', WANDB_MODE=os.environ.get('WANDB_MODE', 'offline'),
        UNIRETARGET_RUN_TAG=os.environ.get('UNIRETARGET_RUN_TAG',
            'chord100m-'+hashlib.sha256(f'{socket.gethostname()}:{root}'.encode()).hexdigest()[:10]))
    for key, value in variables.items():
        command += ['-e', f'{key}={value}']
    # Forward credentials through Docker's environment, never command text or files in Git.
    for key in ('WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT', 'WANDB_BASE_URL'):
        if key in os.environ:
            command += ['-e', key]
    command += [a.image, 'python', str(PACKAGE / 'container_entry.py'), '--run-root', str(root),
                '--episode', str(a.episode), '--seed', str(a.seed), '--phase', a.phase, '--upper', a.upper]
    if a.prepare_only:
        command.append('--prepare-only')
    if a.dry_run:
        print(shlex.join(command))
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
