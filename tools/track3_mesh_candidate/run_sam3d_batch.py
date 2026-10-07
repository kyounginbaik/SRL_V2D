"""Generate one SAM3D mesh per distinct public Track 3 object."""
import json
import os
import traceback
from pathlib import Path

import torch
from v2d.sam3d.lib.image_to_mesh import image_to_mesh

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public')
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam3d'


def main():
    errors = {}
    for item in json.loads((ROOT / 'objects.json').read_text()):
        name, episode, frame = item['name'], item['episode'], item['frame']
        src = SOURCE / f'episode_{episode:06d}'
        image = src / f'frames/{frame:06d}.png'
        mask = (src / f'masks/1/{frame:06d}.png' if item['mask'] == 'existing'
                else ROOT / 'inputs' / name / f'mask_{frame:06d}.png')
        out = ROOT / 'meshes' / name
        out.mkdir(parents=True, exist_ok=True)
        mesh = out / 'generated.glb'
        if mesh.is_file() and mesh.stat().st_size > 1000:
            print(f'SKIP {name}: {mesh}', flush=True)
            continue
        try:
            print(f'START {name}: episode={episode}, image={image}, mask={mask}', flush=True)
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f'{image} or {mask}')
            image_to_mesh(str(image), str(mask), str(mesh),
                          str(out / 'transform.json'), str(out / 'intrinsics.json'),
                          WEIGHTS, seed=42)
            if not mesh.is_file() or mesh.stat().st_size <= 1000:
                raise ValueError('mesh output missing or empty')
            print(f'DONE {name}: {mesh.stat().st_size} bytes', flush=True)
        except Exception as exc:
            errors[name] = repr(exc)
            print(f'FAILED {name}: {exc}', flush=True)
            traceback.print_exc()
        torch.cuda.empty_cache()
    (ROOT / 'inference_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
