#!/usr/bin/env python3
"""Create a self-contained, checksummed, fresh run from the portable recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

RECIPE = Path(__file__).resolve().parent
SOURCE = RECIPE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Directory containing frozen/ and priors/ from the episode-1 bundle')
    parser.add_argument('--run-root', type=Path, required=True, help='New, nonexistent output directory')
    args = parser.parse_args()
    data, root = args.data_root.resolve(), args.run_root.resolve()
    if root.exists():
        parser.error(f'Run root already exists: {root}; full bilevel resume is unsupported')
    if SOURCE == root or SOURCE in root.parents:
        parser.error('Choose a run directory outside the FlashCHORD source tree')
    hashes = json.loads((RECIPE/'data_manifest.json').read_text())
    for name, expected in hashes.items():
        path = data/name
        if not path.is_file() or digest(path) != expected:
            parser.error(f'Missing or mismatched data file: {path}')
    # Catch incomplete git-lfs clones before copying or allocating a GPU.
    for path in (SOURCE/'src/flash_chord/assets').rglob('*'):
        if path.is_file() and path.stat().st_size < 1024:
            if path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
                parser.error(f'Unmaterialized LFS asset: {path}; run git lfs pull')
    root.mkdir(parents=True)
    shutil.copytree(SOURCE, root/'flash_chord', ignore=shutil.ignore_patterns(
        'experiments', '__pycache__', '.pytest_cache', '.cache', '.venv',
        'outputs', 'logs', 'wandb', '*.egg-info', '.git'))
    for name in hashes:
        dest = root/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data/name, dest)
    base = root/'episode_000001'
    base.mkdir()
    shutil.copytree(RECIPE/'code', base/'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(RECIPE/'cluster', root/'cluster', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(RECIPE/'run.sh', root/'run.sh')
    shutil.copy2(RECIPE/'frozen_manifest.json', base/'frozen_manifest.json')
    shutil.copy2(RECIPE/'data_manifest.json', root/'data_manifest.json')
    for directory in ('locks', 'slurm', 'runs'):
        (base/directory).mkdir()
    for directory in ('reports', 'warp_cache', 'slurm'):
        (root/directory).mkdir()
    for name in ('flash_chord', 'frozen', 'priors', 'warp_cache'):
        (base/name).symlink_to('../'+name, target_is_directory=True)
    source_hashes = {str(p.relative_to(root/'flash_chord')): digest(p)
                     for p in sorted((root/'flash_chord').rglob('*')) if p.is_file()}
    (root/'source_manifest.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    launcher_hashes = {str(p.relative_to(root)): digest(p)
                      for folder in (base/'code', root/'cluster')
                      for p in sorted(folder.rglob('*')) if p.is_file()}
    launcher_hashes['run.sh'] = digest(root/'run.sh')
    (root/'launcher_manifest.json').write_text(json.dumps(launcher_hashes, indent=2)+'\n')
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    (root/'reports/preparation.json').write_text(json.dumps(dict(
        source=str(SOURCE), git_revision=revision, data_root=str(data), run_root=str(root),
        source_manifest_sha256=digest(root/'source_manifest.json'),
        launcher_manifest_sha256=digest(root/'launcher_manifest.json'),
        data_manifest_sha256=digest(root/'data_manifest.json')), indent=2)+'\n')
    print(f'Prepared {root}\nCPU preflight: bash {root}/run.sh preflight\n'
          f'Full training: GPU_DEVICE=0 bash {root}/run.sh full 43')


if __name__ == '__main__':
    main()
