#!/usr/bin/env python3
"""ep33 one-row comparison: the original RGB once, then the MuJoCo half of each mink_video.py render (RGB | MuJoCo,
2028x760, 20 fps), one column per reconstruction. Stills at the given times.
usage: ep33_row_compare.py OUT_MP4 "LABEL=VIDEO" ... [--scale 0.75] [--crf 26] [--stills 9.25,12.95] [--title TEXT]"""
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
    s = float(arg('--scale', 0.75)); stills = [float(t) for t in arg('--stills', '9.25,12.95').split(',')]
    title = arg('--title', 'ep33')
    caps = [cv2.VideoCapture(p) for _, p in cols]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 20.0
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))
    w0, h0 = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)); half = w0 // 2
    pw, ph = int(half * s) // 2 * 2, int(h0 * s) // 2 * 2; top, bar = 30, 30
    W, H = pw * (1 + len(cols)), top + bar + ph
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', arg('--crf', '26'),
                             '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    want = {int(round(t * fps)): t for t in stills}
    put = lambda im, txt, x, y, sc=0.6, col=(235, 235, 235): cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    for f in range(n):
        ims = [c.read()[1] for c in caps]
        canvas = np.full((H, W, 3), 18, np.uint8)
        put(canvas, f'{title}   f{f} ({f / fps:.2f}s)', 8, 21)
        canvas[top + bar:, 0:pw] = cv2.resize(ims[0][:, :half], (pw, ph)); put(canvas, 'original RGB', 8, top + 21, 0.6, (200, 200, 200))
        for k, ((lab, _), im) in enumerate(zip(cols, ims)):
            x0 = pw * (k + 1)
            canvas[top + bar:, x0:x0 + pw] = cv2.resize(im[:, half:], (pw, ph))
            put(canvas, lab, x0 + 8, top + 21, 0.6, (120, 210, 255) if k == len(cols) - 1 else (235, 235, 235))
            cv2.line(canvas, (x0, top), (x0, H), (60, 60, 60), 1)
        proc.stdin.write(canvas.tobytes())
        if f in want:
            cv2.imwrite(out.replace('.mp4', f'_t{want[f]}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    proc.stdin.close(); proc.wait(); print('wrote', out, n, 'frames', W, 'x', H)


if __name__ == '__main__':
    main()
