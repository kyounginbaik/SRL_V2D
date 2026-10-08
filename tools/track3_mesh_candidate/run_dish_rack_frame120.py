"""Generate single-view dish-rack candidates with a tracked frame-120 mask."""
import json
from pathlib import Path

from v2d.sam3d.lib.image_to_mesh import image_to_mesh

ROOT = Path('/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark')
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public/episode_000012')
IMAGE = SOURCE / 'frames/000120.png'
MASK = ROOT / 'mvsam3d/dish_rack_tracking/masks/1/000120.png'
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam3d'


def main():
    if not IMAGE.is_file() or not MASK.is_file():
        raise FileNotFoundError(f'{IMAGE} or {MASK}')
    for seed in (17, 42):
        out = ROOT / 'candidates/plastic_dish_rack' / f'ep000012_f000120_s{seed}_tracked'
        out.mkdir(parents=True, exist_ok=True)
        mesh = out / 'generated.glb'
        if not mesh.is_file():
            print('START', mesh, flush=True)
            image_to_mesh(str(IMAGE), str(MASK), str(mesh),
                          str(out / 'transform.json'),
                          str(out / 'intrinsics.json'), WEIGHTS, seed=seed)
        (out / 'source.json').write_text(json.dumps({
            'object': 'plastic_dish_rack', 'episode': 12, 'frame': 120,
            'seed': seed, 'image': str(IMAGE), 'mask': str(MASK)}, indent=2) + '\n')
        print('DONE', mesh, mesh.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
