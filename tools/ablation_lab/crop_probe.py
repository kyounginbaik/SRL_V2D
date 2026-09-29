"""Single-frame HaMeR crop ablation inside the existing v2d_hamer container.

The only runtime adaptation is CPU device selection in a private in-memory copy
of the baseline helper. Weights, preprocessing and predictions are unchanged.
This is a diagnostic probe, not a new full-sequence reconstruction.
"""
import json
from pathlib import Path
import types

import numpy as np
from PIL import Image
import torch


def main():
    torch.set_num_threads(2)
    source = Path('/repo/reconstruction/modules/v2d_hamer/lib/_hamer.py')
    code = source.read_text()
    assert code.count('"cuda"') == 3, 'Review baseline device handling before adapting it'
    module = types.ModuleType('hamer_cpu_probe')
    exec(compile(code.replace('"cuda"', '"cpu"'), str(source), 'exec'), module.__dict__)
    root = Path('/input')
    frame = 244
    image = np.array(Image.open(root / f'frames/{frame:06d}.png').convert('RGB'))
    h, w = image.shape[:2]
    original = json.loads((root / f'hamer/3/{frame:06d}.json').read_text())
    detections = json.loads((root / f'wilor_raw/{frame:06d}.json').read_text())
    detection = next(d for d in detections if not d['is_right'])
    bbox = detection['bbox'];center = [(bbox['x0'] + bbox['x1']) / 2, (bbox['y0'] + bbox['y1']) / 2]
    size = max(bbox['x1'] - bbox['x0'], bbox['y1'] - bbox['y0']) * 1.7
    original_bbox = original['bbox']
    crops = {
        'baseline_sam2_crop': ((original_bbox['x0'] + original_bbox['x1']) / 2,
                               (original_bbox['y0'] + original_bbox['y1']) / 2,
                               original_bbox['x1'] - original_bbox['x0']),
        'wilor_detector_crop': (*center, size),
    }
    model, cfg = module.get_model('/weights/hamer')
    intr = json.loads((root / 'intrinsics_stable.json').read_text())
    output = Path('/output');output.mkdir(exist_ok=True)
    report = {'episode': 41, 'frame': frame, 'side': 'left', 'device': 'cpu', 'model': 'same baseline HaMeR weights',
              'scope': 'Two fresh forward passes on one frame. No depth alignment, no gsplat, no IK. This does not establish sequence-level accuracy.', 'results': {}}
    for name, (cx, cy, side) in crops.items():
        prediction = module.run_hamer(model, cfg, image, cx, cy, side, is_right=False)
        t = prediction['pred_cam_t_full'];f = prediction['scaled_focal_length']
        u = f * t[0] / t[2] + w / 2;v = f * t[1] / t[2] + h / 2;z = t[2] * intr['fx'] / f
        calibrated = np.array([(u - intr['cx']) * z / intr['fx'], (v - intr['cy']) * z / intr['fy'], z])
        report['results'][name] = {'crop_cx_cy_size': [cx, cy, side], 'calibrated_cam_t': calibrated.tolist(),
                                  'camera_distance_m': float(np.linalg.norm(calibrated)), 'pred_cam': prediction['pred_cam'].tolist()}
        np.savez(output / (name + '.npz'), **prediction, calibrated_cam_t=calibrated)
        print(name, report['results'][name], flush=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
