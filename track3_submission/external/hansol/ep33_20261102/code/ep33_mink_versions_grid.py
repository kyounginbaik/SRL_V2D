#!/usr/bin/env python3
"""ep33 Sharpa Mink IK review videos of successive reconstruction versions, side by side. Each input is a mink_video.py
render (RGB | MuJoCo, 2028x760, 20 fps); the grid shows the RGB once and the MuJoCo half of every version:
row 1 = HGA versions, row 2 = HGB versions (RGB repeated). Stills at the given times.
usage: ep33_mink_versions_grid.py OUT_MP4 "LABEL=HGA_VIDEO,HGB_VIDEO" ... [--scale 0.375] [--stills 0.5,8.8,12.4]"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    out = sys.argv[1]
    cols = [a.split('=', 1) for a in sys.argv[2:] if '=' in a and not a.startswith('--')]
    cols = [(lab, paths.split(',')) for lab, paths in cols]
    s = float(arg('--scale', 0.375)); stills = [float(t) for t in arg('--stills', '0.5,8.8,12.4').split(',')]
    caps = [[cv2.VideoCapture(p) for p in paths] for _, paths in cols]
    fps = caps[0][0].get(cv2.CAP_PROP_FPS) or 20.0
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for row in caps for c in row))
    w0, h0 = int(caps[0][0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0][0].get(cv2.CAP_PROP_FRAME_HEIGHT)); half = w0 // 2
    pw, ph = int(half * s) // 2 * 2, int(h0 * s) // 2 * 2; bar = 26
    W, H = pw * (1 + len(cols)), (ph + bar) * 2 + 24
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', arg('--crf', '30'),
                             '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    want = {int(round(t * fps)): t for t in stills}
    put = lambda im, txt, x, y, sc=0.5, col=(235, 235, 235): cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    for f in range(n):
        frames = [[c.read()[1] for c in [row[k] for row in caps]] for k in range(2)]  # [HGA row, HGB row] x versions
        canvas = np.full((H, W, 3), 18, np.uint8)
        put(canvas, f'ep33 f{f} ({f / fps:.2f}s)   left: RGB   then Mink IK render per reconstruction version (top HGA, bottom HGB)', 8, 17)
        for r, (row, name) in enumerate(zip(frames, ('HGA', 'HGB'))):
            y0 = 24 + r * (ph + bar)
            rgb = cv2.resize(row[0][:, :half], (pw, ph)); canvas[y0 + bar:y0 + bar + ph, 0:pw] = rgb
            put(canvas, f'RGB ({name} row)', 6, y0 + 18, 0.48, (200, 200, 200))
            for k, ((lab, _), im) in enumerate(zip(cols, row)):
                x0 = pw * (k + 1)
                canvas[y0 + bar:y0 + bar + ph, x0:x0 + pw] = cv2.resize(im[:, half:], (pw, ph))
                put(canvas, f'{name} {lab}', x0 + 6, y0 + 18, 0.48, (120, 210, 255) if k == len(cols) - 1 else (235, 235, 235))
            for k in range(1, len(cols) + 1):
                cv2.line(canvas, (pw * k, y0), (pw * k, y0 + bar + ph), (60, 60, 60), 1)
        proc.stdin.write(canvas.tobytes())
        if f in want:
            cv2.imwrite(out.replace('.mp4', f'_t{want[f]}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    proc.stdin.close(); proc.wait(); print('wrote', out, n, 'frames', W, 'x', H)


if __name__ == '__main__':
    main()
