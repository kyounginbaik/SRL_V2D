#!/usr/bin/env python3
"""ep33 frame-0 rest roll, side by side on the RGB frames: two bundles' spoon (mesh outline, long axis and a body-fixed
cross tick) over the SAM2 spoon mask outline (white), the GT-free marker-ball detections (yellow) and the bundle's projected
ball centres (cyan). Per panel: silhouette IoU and the median ball distance (px), as in K33's imgev. Top: full frame,
bottom: zoom on the spoon. GT-free.
usage: ep33_f0roll_video.py EP_DIR_A LABEL_A EP_DIR_B LABEL_B OUT_MP4 [--frames 0-80] [--fps 10]"""
import json
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import trimesh

C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
BC = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load(ep):
    z = np.load(f'{ep}/object_poses.npz', allow_pickle=False)
    T = z['camera_from_object'][:, list(z['names']).index('wooden_spoon')]
    m = trimesh.load(f'{ep}/meshes/wooden_spoon.obj', force='mesh'); outer = max(m.split(only_watertight=False), key=lambda p: p.area)
    c0 = np.asarray(m.vertices) - np.asarray(m.vertices).mean(0); _, U = np.linalg.eigh(c0.T @ c0 / len(c0))
    return T, np.asarray(outer.vertices), np.asarray(outer.faces), np.asarray(m.center_mass), U


def main():
    a_ep, a_lab, b_ep, b_lab, out = sys.argv[1:6]
    f0, f1 = (int(x) for x in arg('--frames', '0-80').split('-')); fps = float(arg('--fps', 10))
    K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
    pe = json.load(open(BC))['per_episode_centres']; balls = np.array([np.median([pe[e][k] for e in pe], 0) for k in ('near', 'far', 'tip')])
    det = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
    A, B = load(a_ep), load(b_ep)
    cap = cv2.VideoCapture(f'{C}/rgb.mp4'); frames = []
    while len(frames) <= f1:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    proj = lambda P: np.stack([fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy], 1)

    def panel(img, t, S, mask, dets, col):
        T, v, f, com, U = S; Tt = T[t]; p = v @ Tt[:3, :3].T + Tt[:3, 3]; uv = proj(p).round().astype(np.int32)
        sil = np.zeros((H, W), np.uint8)
        for tri in f:  # one convex fill per triangle (a single fillPoly over all triangles drops overlaps)
            cv2.fillConvexPoly(sil, uv[tri], 1)
        iou = (sil.astype(bool) & mask).sum() / max((sil.astype(bool) | mask).sum(), 1)
        out = img.copy()
        cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(out, cs, -1, col, 2, cv2.LINE_AA)
        cm, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(out, cm, -1, (255, 255, 255), 1, cv2.LINE_AA)
        c = (Tt[:3, :3] @ com + Tt[:3, 3])[None]; ax = np.vstack([c, c + 0.12 * (Tt[:3, :3] @ U[:, 2])[None], c + 0.04 * (Tt[:3, :3] @ U[:, 1])[None]])
        q = proj(ax).round().astype(int); cv2.line(out, tuple(q[0]), tuple(q[1]), col, 1, cv2.LINE_AA); cv2.line(out, tuple(q[0]), tuple(q[2]), col, 3, cv2.LINE_AA)
        pb = proj(balls @ Tt[:3, :3].T + Tt[:3, 3])
        for u in pb:
            cv2.circle(out, tuple(np.round(u).astype(int)), 5, (255, 255, 0), 1, cv2.LINE_AA)
        for u in dets:
            cv2.circle(out, tuple(np.round(u).astype(int)), 3, (0, 220, 255), -1, cv2.LINE_AA)
        bpx = float(np.median(np.min(np.linalg.norm(pb[:, None] - dets[None], axis=2), 1))) if len(dets) >= 2 else np.nan
        return out, iou, bpx, uv

    Wp = 640; Hp = int(H * Wp / W) // 2 * 2; Z = 320   # libx264 needs even dimensions
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{2 * Wp}x{Hp + Z + 30}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', arg('--crf', '24'), '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    stats = []
    for t in range(f0, min(f1 + 1, len(frames))):
        mask = cv2.imread(f'{C}/objects/wooden_spoon/masks/1/{t:06d}.png', cv2.IMREAD_UNCHANGED) > 0
        d = det.get(str(t)) if isinstance(det, dict) else (det[t] if len(det) > t else None)
        dets = np.array([x['uv'] for x in d]) if d else np.zeros((0, 2))
        cols, row_top, row_zoom = [(80, 220, 80), (255, 140, 60)], [], []   # BGR: green, blue
        ys, xs = np.nonzero(mask)
        ctr = (np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([W / 2, H / 2])).astype(int)
        half = 140
        for S, lab, col in ((A, a_lab, cols[0]), (B, b_lab, cols[1])):
            im, iou, bpx, _ = panel(frames[t], t, S, mask, dets, col)
            top = cv2.resize(im, (Wp, Hp))
            cv2.putText(top, f'{lab}   IoU {iou:.3f}   balls {bpx:4.1f}px', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            x0, y0 = np.clip(ctr - half, 0, [W - 2 * half, H - 2 * half]); crop = im[y0:y0 + 2 * half, x0:x0 + 2 * half]
            row_top.append(top); row_zoom.append(cv2.resize(crop, (Wp, Z)) if crop.size else np.zeros((Z, Wp, 3), np.uint8))
            stats.append((t, lab, iou, bpx))
        bar = np.full((30, 2 * Wp, 3), 24, np.uint8)
        cv2.putText(bar, f'ep33 f{t} ({t / 20:.2f}s)  spoon static until ~f32   green/blue = bundle spoon, white = SAM2 mask, yellow = detected balls, cyan = mesh balls',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        frame = np.vstack([bar, np.hstack(row_top), np.hstack(row_zoom)]); proc.stdin.write(frame.tobytes())
        if t in (0, 12, 24, 40):
            cv2.imwrite(out.replace('.mp4', f'_f{t:03d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, int(arg('--jpg', 90))])
    proc.stdin.close(); proc.wait()
    s = np.array([(r[2], r[3]) for r in stats], float).reshape(-1, 2, 2)
    for k, lab in enumerate((a_lab, b_lab)):
        for lo, hi in ((0, 25), (26, 32), (33, 60)):
            sel = [i for i, r in enumerate(range(f0, f0 + len(s))) if lo <= r <= hi]
            if sel:
                print(f'{lab:>12s} f{lo}-{hi}: IoU {np.nanmean(s[sel, k, 0]):.4f}  balls {np.nanmedian(s[sel, k, 1]):.2f}px')
    print('wrote', out)


if __name__ == '__main__':
    main()
