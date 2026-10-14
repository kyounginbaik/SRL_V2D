"""Prepare the portable bundle, then execute the same smoke/full launch chain."""
from pathlib import Path
import argparse
import subprocess
import sys

from prepare import prepare

p = argparse.ArgumentParser()
p.add_argument('--run-root', type=Path, required=True)
p.add_argument('--episode', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--phase', required=True)
p.add_argument('--upper', choices=('gaussian','flashsac'), default='gaussian')
p.add_argument('--prepare-only', action='store_true')
a = p.parse_args()
prepare(a.run_root, a.upper)
if not a.prepare_only:
    subprocess.run([sys.executable, str(a.run_root / f'episode_{a.episode:06d}' / 'code/gpu_entry.py'),
                    '--seed', str(a.seed), '--phase', a.phase], check=True)
