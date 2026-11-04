#!/usr/bin/env python3
"""RGB crops of the ep32 spoon for manual front/back annotation (RGB clock, same frames as annot_json marks): a square
crop around the SAM2 spoon mask (or the previous frame's box when the mask is missing), JPEG, plus frames.json with the
crop box and the 11/02 mark of each frame. No reconstruction is drawn.
usage: ep33_annot_crops.py OUT_DIR F0-F1[,F0-F1...] STEP [ANN_JSON]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

C = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032')


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); st = int(sys.argv[3])
    frames = sorted({f for r in sys.argv[2].split(',') for f in range(int(r.split('-')[0]), int(r.split('-')[1]) + 1, st)})
    ann = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else {}
    cap = cv2.VideoCapture(str(C / 'rgb.mp4')); t = 0; last = None; rows = []
    while t <= frames[-1]:
        ok, im = cap.read()
        if not ok:
            break
        if t in frames:
            mp = C / 'objects/wooden_spoon/masks/1' / f'{t:06d}.png'
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) > 0 if mp.exists() else None
            if m is not None and m.any():
                ys, xs = np.nonzero(m); box = (xs.min(), ys.min(), xs.max(), ys.max())
            else:
                box = last
            x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = int(max(x1 - x0, y1 - y0) * 0.62 + 40)
            X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half)); X1, Y1 = int(min(im.shape[1], cx + half)), int(min(im.shape[0], cy + half))
            crop = cv2.resize(im[Y0:Y1, X0:X1], (440, int(440 * (Y1 - Y0) / max(1, X1 - X0)) // 2 * 2))
            cv2.imwrite(str(out / f'f{t:03d}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 84])
            rows.append(dict(frame=t, time_s=round(t / 20, 2), box=[X0, Y0, X1, Y1], prev=ann.get(str(t))))
            last = box
        t += 1
    (out / 'frames.json').write_text(json.dumps(rows, indent=1)); print('wrote', len(rows), 'crops to', out)


if __name__ == '__main__':
    main()
