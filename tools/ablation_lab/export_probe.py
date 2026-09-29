"""Publish the measured, one-frame CPU crop probe as separate evidence."""
import json
from pathlib import Path

import cv2
import numpy as np

from .build import REPO, write_json


def export_probe(site=REPO / '.ablation_lab/site'):
    root = REPO / '.ablation_lab/crop_probe'
    if not (root / 'report.json').exists():
        return
    report = json.loads((root / 'report.json').read_text())
    source_root = Path('/data1/video_to_data_challenge/runs/recon/mesh/episode_000041')
    image = cv2.imread(str(source_root / 'frames/000244.png'))
    intr = json.loads((source_root / 'intrinsics_stable.json').read_text())
    faces = np.asarray(json.loads((site / 'hand_faces.json').read_text()))
    for name, result in report['results'].items():
        pred = np.load(root / (name + '.npz'), allow_pickle=False)
        vertices = pred['pred_vertices'] + pred['calibrated_cam_t']
        uv = vertices[:, :2] / vertices[:, 2:]
        uv = uv * [intr['fx'], intr['fy']] + [intr['cx'], intr['cy']]
        canvas = image.copy();overlay = image.copy()
        for face in faces[np.argsort(vertices[faces, 2].mean(1))[::-1]]:
            if (vertices[face, 2] <= .01).any() or np.abs(uv[face]).max() > 100000:
                continue
            cv2.fillConvexPoly(overlay, uv[face].astype(np.int32), (190, 215, 90), lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, .55, canvas, .45, 0)
        cx, cy, size = result['crop_cx_cy_size']
        cv2.rectangle(canvas, (round(cx-size/2), round(cy-size/2)), (round(cx+size/2), round(cy+size/2)), (100, 245, 195), 5)
        canvas = cv2.resize(canvas, (760, round(image.shape[0] * 760 / image.shape[1])))
        destination = site / (name + '.jpg');cv2.imwrite(str(destination), canvas)
        result['image'] = '/data/' + destination.name
    write_json(site / 'crop_probe.json', report)
    ep_path = site / 'episode_000041/episode.json';episode = json.loads(ep_path.read_text())
    episode['crop_probe'] = report
    message = 'Fresh one-frame HaMeR crop ablation at frame 244: SAM2 crop 12.53 m → WiLoR detector crop 0.62 m. Same weights, CPU forward passes, before depth alignment; sequence-level improvement is not established.'
    if message not in episode['findings']:
        episode['findings'].insert(0, message)
    write_json(ep_path, episode)
    report_path = site / 'REPORT.md';text = report_path.read_text()
    marker = '## Fresh HaMeR crop ablation'
    if marker not in text:
        report_path.write_text(text + '\n' + marker + '\n\n' + message + '\n\nThe original SAM2 crop has side 1101.6 pixels; the detector crop has side 457.3 pixels (the same 1.7 expansion). The baseline CPU estimate is 12.53384 m, matching the saved GPU estimate 12.53385 m. Detector-crop estimate is 0.61784 m. No downstream depth alignment, gsplat or retargeting was applied to this probe.\n')
    print('Published one-frame crop evidence')


if __name__ == '__main__':
    export_probe()
