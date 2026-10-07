"""Generate wooden-spoon SAM3D candidates from evaluation videos only."""
from pathlib import Path
import json
import os
import traceback

from v2d.sam3d.lib.image_to_mesh import image_to_mesh

ROOT = Path(os.environ.get('V2D_CHALLENGE_RECON_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction'))
OUT = ROOT / 'spoon_mesh_candidates'
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam3d'
CANDIDATES = [(35, 0, 42), (35, 240, 42), (36, 0, 42), (37, 0, 42), (35, 0, 17)]


def main():
    errors = {}
    for episode, frame, seed in CANDIDATES:
        name = f'ep{episode:06d}_f{frame:06d}_s{seed}'
        src = ROOT / f'evaluation/episode_{episode:06d}'
        image = src / f'frames/{frame:06d}.png'
        mask = src / f'masks/1/{frame:06d}.png'
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        mesh = out / 'generated.glb'
        if mesh.is_file() and mesh.stat().st_size > 1000:
            print('SKIP', name, flush=True)
            continue
        try:
            print('START', name, flush=True)
            image_to_mesh(str(image), str(mask), str(mesh), str(out / 'transform.json'),
                          str(out / 'intrinsics.json'), WEIGHTS, seed=seed)
            if not mesh.is_file() or mesh.stat().st_size <= 1000:
                raise ValueError('empty mesh')
            print('DONE', name, mesh.stat().st_size, flush=True)
        except Exception as exc:
            errors[name] = repr(exc)
            traceback.print_exc()
    (OUT / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
