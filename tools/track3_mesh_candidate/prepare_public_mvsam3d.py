"""Build MV-SAM3D inputs from public video frames and tracked object masks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

PUBLIC_RUNS = Path('/data1/SRL_V2D/challenge_reconstruction/public')
BENCHMARK = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                                '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('object_name')
    parser.add_argument('episode', type=int)
    parser.add_argument('frames', help='comma-separated frame numbers')
    parser.add_argument('--mask-slot', type=int, default=1)
    parser.add_argument('--tag', default='5v')
    args = parser.parse_args()

    frames = [int(value) for value in args.frames.split(',')]
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError('provide at least two distinct frames')
    source = PUBLIC_RUNS / f'episode_{args.episode:06d}'
    out = BENCHMARK / 'mvsam3d' / f'{args.object_name}_{args.tag}' / 'input'
    images = out / 'images'
    masks = out / args.object_name
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        name = f'{frame:06d}.png'
        image = source / 'frames' / name
        mask_path = source / 'masks' / str(args.mask_slot) / name
        if not image.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f'{image} or {mask_path}')
        target = images / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(image)
        mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        if mask.sum() < 200:
            raise ValueError(f'{mask_path}: empty mask')
        rgba = np.full((*mask.shape, 4), 255, dtype=np.uint8)
        rgba[..., 3] = mask.astype(np.uint8) * 255
        Image.fromarray(rgba, 'RGBA').save(masks / name)
    (out / 'input_manifest.json').write_text(json.dumps({
        'episode': args.episode, 'object': args.object_name,
        'frames': frames, 'mask_slot': args.mask_slot,
        'source': 'public video and tracked masks'}, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
