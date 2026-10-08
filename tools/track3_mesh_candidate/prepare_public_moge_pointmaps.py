"""Convert saved MoGe depth into pointmaps accepted by MV-SAM3D.

MV-SAM3D calls the input flag ``--da3_output``. The file generated here uses
MoGe predictions from the public video pipeline, not Depth Anything 3.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path,
                        help='MV-SAM3D input directory containing input_manifest.json')
    parser.add_argument('--process-res', type=int, default=504)
    args = parser.parse_args()
    manifest_path = args.input_dir / 'input_manifest.json'
    if not manifest_path.is_file():
        manifest_path = args.input_dir.parent / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    episode = SOURCE / f"episode_{manifest['episode']:06d}"
    depths, pointmaps, intrinsics, filenames = [], [], [], []
    for frame in manifest['frames']:
        name = f'{frame:06d}'
        raw = np.asarray(Image.open(episode / 'depth' / f'{name}.png'), dtype=np.float32)
        depth = 65535.0 / np.maximum(raw, 1.0) - 1.0
        calib = json.loads((episode / 'intrinsics' / f'{name}.json').read_text())
        h, w = depth.shape
        scale = args.process_res / max(h, w)
        new_w, new_h = round(w * scale), round(h * scale)
        depth = np.asarray(Image.fromarray(depth, 'F').resize(
            (new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32)
        depth = np.maximum(depth, 0.01)
        fx, fy = calib['fx'] * new_w / w, calib['fy'] * new_h / h
        cx, cy = calib['cx'] * new_w / w, calib['cy'] * new_h / h
        yy, xx = np.mgrid[:new_h, :new_w].astype(np.float32)
        pointmap = np.stack(((xx - cx) / fx * depth,
                             (yy - cy) / fy * depth, depth), axis=0)
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        depths.append(depth)
        pointmaps.append(pointmap)
        intrinsics.append(k)
        filenames.append(str(args.input_dir / 'images' / f'{name}.png'))
        print(name, float(np.median(depth)), pointmap.shape, flush=True)
    out = args.input_dir / 'moge_pointmaps.npz'
    np.savez_compressed(out, depth=np.stack(depths),
                        pointmaps_sam3d=np.stack(pointmaps),
                        intrinsics=np.stack(intrinsics),
                        image_files=np.array(filenames),
                        process_res=args.process_res)
    print(out, flush=True)


if __name__ == '__main__':
    main()
