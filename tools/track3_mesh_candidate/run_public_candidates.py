"""Generate additional video-only SAM3D meshes for the public benchmark."""
import json
import os
import traceback
from pathlib import Path

from v2d.sam3d.lib.image_to_mesh import image_to_mesh

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public')
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam3d'

# The first alternate seed covers every object. Extra views focus on the
# objects that dominate the current weighted CD-O error.
EXTRA = {
    'white_pot': [(60, 42), (150, 42), (150, 17)],
    'water_pitcher': [(120, 42), (150, 42), (180, 42), (120, 17),
                      (120, 123), (120, 777)],
    'blue_cup': [(60, 42), (90, 42)],
    'wooden_piece_1': [(0, 123), (0, 777)],
    'wooden_piece_2': [(0, 123)],
    'plastic_dish_rack': [(0, 123), (0, 777), (0, 2026)],
    'mini_dust_pan': [(215, 123), (215, 777)],
    'beige_cup': [(0, 123)],
}


def main():
    errors = {}
    for item in json.loads((ROOT / 'objects.json').read_text()):
        name, episode, initial = item['name'], item['episode'], item['frame']
        for frame, seed in [(initial, 17), *EXTRA.get(name, [])]:
            tag = f'ep{episode:06d}_f{frame:06d}_s{seed}'
            out = ROOT / 'candidates' / name / tag
            mesh = out / 'generated.glb'
            if mesh.is_file() and mesh.stat().st_size > 1000:
                print('SKIP', name, tag, flush=True)
                continue
            image = SOURCE / f'episode_{episode:06d}/frames/{frame:06d}.png'
            mask = (SOURCE / f'episode_{episode:06d}/masks/1/{frame:06d}.png'
                    if item['mask'] == 'existing' else
                    ROOT / 'inputs' / name / f'mask_{initial:06d}.png')
            if item['mask'] != 'existing' and frame != initial:
                errors[f'{name}/{tag}'] = 'alternate frame lacks a mask'
                continue
            try:
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f'{image} or {mask}')
                out.mkdir(parents=True, exist_ok=True)
                print('START', name, tag, flush=True)
                image_to_mesh(str(image), str(mask), str(mesh),
                              str(out / 'transform.json'), str(out / 'intrinsics.json'),
                              WEIGHTS, seed=seed)
                if not mesh.is_file() or mesh.stat().st_size < 1000:
                    raise ValueError('missing or empty mesh')
                (out / 'source.json').write_text(json.dumps({
                    'name': name, 'episode': episode, 'frame': frame,
                    'seed': seed, 'image': str(image), 'mask': str(mask)}, indent=2) + '\n')
                print('DONE', name, tag, mesh.stat().st_size, flush=True)
            except Exception as exc:
                errors[f'{name}/{tag}'] = repr(exc)
                traceback.print_exc()
    (ROOT / 'candidate_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
