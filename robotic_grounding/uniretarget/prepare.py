"""Materialize relocatable inputs and launchers without altering source data."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE.parent / 'flash_chord'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, upper='gaussian'):
    if upper not in ('gaussian', 'flashsac', 'wrench_hypergradient'):
        raise ValueError(f'Unknown upper optimizer: {upper}')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_hashes = {str(p.relative_to(SOURCE)): digest(p)
                         for folder in ('src', 'scripts', 'tests')
                         for p in sorted((SOURCE / folder).rglob('*'))
                         if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}
        fingerprint = {str(p.relative_to(PACKAGE)): digest(p)
                       for p in PACKAGE.rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts and p.name != '.gitignore'}
        stamp = root / 'prepared.json'
        if stamp.exists():
            previous = json.loads(stamp.read_text())
            if previous['package'] != fingerprint or previous['source'] != source_hashes or previous.get('upper') != upper:
                raise RuntimeError('Code/input bundle changed: choose a new --run-root; existing runs are preserved')
            print(f'Already prepared: {root}')
            return
        (root / 'reports').mkdir(exist_ok=True)
        (root / 'warp_cache').mkdir(exist_ok=True)
        for name, target in (('flash_chord', SOURCE), ('priors', PACKAGE / 'priors')):
            link = root / name
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
        manifests = json.loads((PACKAGE / 'input_manifest.json').read_text())
        for entry in manifests:
            ep = entry['episode']
            source = PACKAGE / 'inputs' / ep
            for relative, expected in {'reference.parquet': entry['runtime_sha256'], **entry['asset_sha256']}.items():
                if digest(source / relative) != expected:
                    raise ValueError(f'Input checksum mismatch: {source / relative}; run git lfs pull')
            frozen = root / 'frozen' / ep
            shutil.copytree(source, frozen, dirs_exist_ok=True)
            table = pq.read_table(frozen / 'reference.parquet')
            row = table.to_pylist()[0]
            for key in ('object_mesh_paths', 'object_urdf_paths'):
                row[key] = [str((frozen / value).resolve()) for value in row[key]]
                if not all(Path(value).is_file() for value in row[key]):
                    raise FileNotFoundError(f'Missing relocated asset in {ep}')
            pq.write_table(pa.Table.from_pylist([row], schema=table.schema), frozen / 'reference.parquet')
            entry['runtime_sha256'] = digest(frozen / 'reference.parquet')
            base = root / ep
            for folder in ('code', 'slurm', 'locks'):
                (base / folder).mkdir(parents=True, exist_ok=True)
            for name in ('flash_chord', 'frozen', 'priors', 'warp_cache'):
                link = base / name
                if not link.exists():
                    link.symlink_to(root / name, target_is_directory=True)
            (base / 'frozen_manifest.json').write_text(json.dumps([entry], indent=2)+'\n')
            for template in (PACKAGE / 'templates').glob('*.py'):
                text = template.read_text().replace('episode_000001', ep).replace('__UPPER_OPTIMIZER__', upper)
                if template.name == 'run_mixed.py':
                    text = text.replace(" config['upper_updates_per_batch']=0", " config['upper_optimizer']='"+upper+"'\n config['upper_updates_per_batch']=0")
                    if upper == 'flashsac':
                        text = text.replace("'+bilevel.upper_updates=0'", "'+bilevel.upper_updates=32','+bilevel.upper_proposal=flashsac'")
                        text = text.replace("'upper_updates_per_batch':0", "'upper_updates_per_batch':32").replace("config['upper_updates_per_batch']=0", "config['upper_updates_per_batch']=32")
                        text = text.replace('Gaussian candidate population with solved lower response for every candidate',
                                            'learned upper FlashSAC Gaussian proposals; solved lower response for every candidate')
                (base / 'code' / template.name).write_text(text)
        (root / 'source_manifest.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
        stamp.write_text(json.dumps(dict(source=source_hashes, package=fingerprint, upper=upper), indent=2)+'\n')
        print(f'Prepared five episodes at {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--upper', choices=('gaussian','flashsac','wrench_hypergradient'), default='gaussian')
    args=parser.parse_args()
    prepare(args.run_root, args.upper)
