"""Run SAM2 image prediction for four manually boxed public-video objects."""
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from v2d.sam2.lib.image_to_mask import _build_predictor

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
SOURCE = Path('/data1/SRL_V2D/challenge_reconstruction/public')
WEIGHTS = '/data1/SRL_V2D/reconstruction/data/weights/sam2'


def main():
    predictor = _build_predictor(WEIGHTS)
    errors = {}
    for item in json.loads((ROOT / 'objects.json').read_text()):
        if item['mask'] != 'sam2':
            continue
        name, episode, frame = item['name'], item['episode'], item['frame']
        try:
            src = SOURCE / f'episode_{episode:06d}/frames/{frame:06d}.png'
            out = ROOT / 'inputs' / name / f'mask_{frame:06d}.png'
            image = np.asarray(Image.open(src).convert('RGB'))
            box = np.asarray(item['box'], dtype=np.float32)
            with torch.inference_mode():
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            area = int((mask > 0).sum())
            if area < 200 or area > image.shape[0] * image.shape[1] * 0.35:
                raise ValueError(f'{name}: implausible SAM2 mask area {area}')
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(out)
            print(f'{name}: mask area={area}, score={float(scores[0]):.3f}, output={out}', flush=True)
        except Exception as exc:
            errors[name] = repr(exc)
            traceback.print_exc()
    (ROOT / 'sam2_errors.json').write_text(json.dumps(errors, indent=2) + '\n')


if __name__ == '__main__':
    main()
