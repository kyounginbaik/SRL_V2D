"""Preserve the selected run without modifying its original files."""
from pathlib import Path
import hashlib
import json
import shutil

BASE = Path(__file__).resolve().parents[1]
SOURCE = Path('/data3/exaFLOPs/uniretarget_tracklower_20261017/repaired_000037_ilchold/episode_000037')
DEST = BASE / 'source/episode_000037'
TRAIN = Path('runs/pilot/episode_000037/chunk/seed_44/training')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    inventory = []
    selected = []
    for rel in ['flash_chord', 'runtime_assets', 'asset_cache', 'frozen', 'priors', 'code', 'demo', 'reproduction']:
        root = SOURCE / rel
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            relative = p.relative_to(SOURCE)
            if any(x in relative.parts for x in ['__pycache__', '.git', '.pytest_cache', '.ruff_cache', 'wheels']):
                continue
            if relative.parts[:4] == ('flash_chord', 'src', 'flash_chord', 'assets') and len(relative.parts) > 4 and relative.parts[4] in ['meshes', 'urdfs', 'policies', '.cache']:
                continue
            if relative.parts[0] == 'code' and p.suffix == '.mp4':
                continue
            selected.append(relative)
    selected.extend([TRAIN / 'policy_2508800.safetensors', TRAIN / 'state_2508800.safetensors',
                     TRAIN / 'upper_updates.jsonl', TRAIN.parent / 'run_config.json',
                     Path('frozen_manifest.json'), Path('slurm/roundeval_2508800.log'), Path('slurm/train-48591.log')])
    selected.extend(p.relative_to(SOURCE) for p in (SOURCE / TRAIN / 'references').glob('reference_000[0-5].npz'))
    selected.extend(p.relative_to(SOURCE) for p in (SOURCE / TRAIN / 'search_batches').glob('step_*/scores.json'))
    for rel in sorted(set(selected)):
        src, dst = SOURCE / rel, DEST / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha(src)
        if sha(dst) != digest:
            raise RuntimeError(f'Copy changed bytes: {rel}')
        inventory.append(dict(path=str(rel), sha256=digest, bytes=dst.stat().st_size))
    video = Path('/data3/exaFLOPs/epi37.mp4')
    if sha(video) != sha(DEST / 'demo/epi37.mp4'):
        raise RuntimeError('Requested video differs from run video')
    manifest = dict(source=str(SOURCE), snapshot=str(DEST), selected_video=str(video),
                    selected_video_sha256=sha(video), selected_checkpoint=str(TRAIN / 'policy_2508800.safetensors'),
                    selected_reference=str(TRAIN / 'references/reference_0005.npz'), selected_world=28,
                    original_path_strings_preserved=True,
                    note='Inference snapshot plus selected training/evaluation evidence. No claims of portable execution until replay validation.',
                    docker_image='sha256:22d733893d932297a65adfa788d663a42c0778842d96aa20ae2713c07b69971c', files=inventory)
    (BASE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(inventory), bytes=sum(x['bytes'] for x in inventory), manifest=str(BASE / 'source_manifest.json'))))


if __name__ == '__main__':
    main()
