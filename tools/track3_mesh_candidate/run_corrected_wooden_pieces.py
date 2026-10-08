"""Reconstruct both wooden pieces using the corrected tabletop masks."""
import json
import os
from pathlib import Path

from v2d.sam3d.lib.image_to_mesh import image_to_mesh

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public/episode_000023/frames/000000.png')
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam3d'


def main():
    for name in ('wooden_piece_1', 'wooden_piece_2'):
        mask = ROOT / 'inputs' / name / 'mask_000000.png'
        if not mask.is_file():
            raise FileNotFoundError(mask)
        for seed in (42, 17, 123):
            tag = f'ep000023_f000000_s{seed}_maskv2'
            out = ROOT / 'candidates' / name / tag
            mesh = out / 'generated.glb'
            if mesh.is_file() and mesh.stat().st_size > 1000:
                print('SKIP', name, tag, flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            print('START', name, tag, flush=True)
            image_to_mesh(str(SOURCE), str(mask), str(mesh),
                          str(out / 'transform.json'), str(out / 'intrinsics.json'),
                          WEIGHTS, seed=seed)
            if not mesh.is_file() or mesh.stat().st_size < 1000:
                raise ValueError(f'{name}/{tag}: missing or empty mesh')
            (out / 'source.json').write_text(json.dumps({
                'name': name, 'episode': 23, 'frame': 0, 'seed': seed,
                'mask_version': 'corrected_tabletop', 'mask': str(mask)}, indent=2) + '\n')
            print('DONE', name, tag, mesh.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
