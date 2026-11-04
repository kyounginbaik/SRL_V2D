#!/usr/bin/env python3
"""ep33 spoon: correct only the tilt toward / away from the camera in given frame windows, from the stereo-depth 3D line
(handle_depth_line.py output, RGB clock, camera frame). GT-free.
Per RGB frame: elevation e(v) = asin(v . r) of a direction v out of the image plane, r = unit viewing ray through the depth
line centroid. dtilt = e(depth line) - e(spoon long axis) (sign-aligned). The episode's constant offset (median dtilt over
reliable no-contact frames) is removed, the rest is median-filtered over +-2 frames, Gaussian-smoothed (sigma 1.5 frames),
scaled by --scale, and applied only inside --windows with --ramp-frame cosine ramps. The spoon is rotated about
k = axis x r through the point of its long axis nearest to the depth line centroid: the axis stays in the plane through the
viewing ray, so its projected in-image direction is unchanged and only the elevation moves. The per-frame body-frame
correction is mapped to the stored clock (+delta) and applied to world_from_object / camera_from_object of the spoon.
usage: ep33_tiltfix.py BUNDLE EPISODE DEPTH_JSON OUT_EPISODE_DIR --windows 156-190,224-264 [--scale 1.0] [--ramp 4]"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/data1/hansol/mink_S1_20261031/code')
from contact_smooth import DELTA, shift, arg  # noqa: E402


def elev(v, r):
    return np.degrees(np.arcsin(np.clip(v @ r, -1, 1)))


def main():
    bundle, ep, dj, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    scale, ramp = arg('--scale', 1.0), int(arg('--ramp', 4))
    wins = [tuple(int(x) for x in w.split('-')) for w in arg('--windows', '').split(',')]
    E = f'episode_{ep:06d}'
    z = dict(np.load(bundle / E / 'object_poses.npz')); names = [str(n) for n in z['names']]; s = names.index('wooden_spoon')
    F = len(z['camera_from_object']); d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
    mj = json.loads((bundle / E / 'meshes.json').read_text()); ms = trimesh.load(bundle / E / mj['wooden_spoon']['file'], force='mesh', process=False)
    cs = ms.vertices.mean(0); ax = np.linalg.eigh(np.cov((ms.vertices - cs).T))[1][:, -1]
    C = shift(z['camera_from_object'][:, s].astype(float), -d)
    rows = {r['f']: r for r in json.loads(dj.read_text())}

    dt = np.full(F, np.nan); cont = np.zeros(F, bool); geo = {}
    for f in range(F):
        r = rows.get(f)
        if not r or not r.get('reliable'):
            continue
        u, c = np.array(r['line_u']), np.array(r['line_c']); ray = c / np.linalg.norm(c)
        a = C[f, :3, :3] @ ax; a = a if a @ u >= 0 else -a
        dt[f] = elev(u, ray) - elev(a, ray); cont[f] = r['contact']; geo[f] = (c, ray)
    ok = ~np.isnan(dt); bias = float(np.median(dt[ok & ~cont]))
    raw = dt - bias
    med = np.array([np.nanmedian(raw[max(0, f - 2):f + 3]) if ok[max(0, f - 2):f + 3].any() else np.nan for f in range(F)])
    idx = np.arange(F); med = np.interp(idx, idx[~np.isnan(med)], med[~np.isnan(med)])
    sm = gaussian_filter1d(med, 1.5)
    w = np.zeros(F)
    for a0, b0 in wins:
        for f in range(max(0, a0 - ramp), min(F, b0 + ramp + 1)):
            if a0 <= f <= b0:
                w[f] = 1.0
            elif f < a0:
                w[f] = max(w[f], 0.5 - 0.5 * np.cos(np.pi * (f - (a0 - ramp)) / ramp))
            else:
                w[f] = max(w[f], 0.5 - 0.5 * np.cos(np.pi * ((b0 + ramp) - f) / ramp))
    corr = scale * w * sm  # degrees of elevation to add

    B_rgb = np.tile(np.eye(4), (F, 1, 1)); applied = []
    for f in range(F):
        if abs(corr[f]) < 1e-6:
            continue
        T = C[f]; R = T[:3, :3]; a = R @ ax
        c, ray = geo.get(f) or geo[min(geo, key=lambda g: abs(g - f))]
        u_sign = 1.0
        if f in rows and rows[f].get('reliable'):
            u_sign = 1.0 if a @ np.array(rows[f]['line_u']) >= 0 else -1.0
        a = u_sign * a
        k = np.cross(a, ray); k /= np.linalg.norm(k)
        # rotating a about k by +theta moves it toward the ray (raises its elevation)
        Rk = Rot.from_rotvec(k * np.radians(corr[f])).as_matrix()
        if elev(Rk @ a, ray) < elev(a, ray) and corr[f] > 0 or elev(Rk @ a, ray) > elev(a, ray) and corr[f] < 0:
            Rk = Rk.T
        p0 = R @ cs + T[:3, 3]; pc = p0 + a * ((c - p0) @ a)  # point on the spoon axis nearest the depth line centroid
        Tn = np.eye(4); Tn[:3, :3] = Rk @ R; Tn[:3, 3] = Rk @ (T[:3, 3] - pc) + pc
        B_rgb[f] = np.linalg.inv(T) @ Tn; applied.append(f)

    rvb = Rot.from_matrix(B_rgb[:, :3, :3]).as_rotvec(); tvb = B_rgb[:, :3, 3]; g = np.arange(F) + d
    rvs = np.stack([np.interp(g, np.arange(F), rvb[:, k]) for k in range(3)], 1); tvs = np.stack([np.interp(g, np.arange(F), tvb[:, k]) for k in range(3)], 1)
    wfo, cfo = z['world_from_object'].copy(), z['camera_from_object'].copy()
    for gg in range(F):
        B = np.eye(4); B[:3, :3] = Rot.from_rotvec(rvs[gg]).as_matrix(); B[:3, 3] = tvs[gg]
        wfo[gg, s] = (wfo[gg, s].astype(float) @ B).astype(wfo.dtype); cfo[gg, s] = (cfo[gg, s].astype(float) @ B).astype(cfo.dtype)
    z['world_from_object'], z['camera_from_object'] = wfo, cfo
    out.mkdir(parents=True, exist_ok=True); np.savez_compressed(out / 'object_poses.npz', **z)
    rep = dict(episode=ep, windows=wins, ramp=ramp, scale=scale, depth_tilt_offset_deg=round(bias, 2), frames_changed=len(applied),
               correction_deg=dict(max=round(float(np.abs(corr).max()), 2), mean_in_windows=[round(float(corr[a0:b0 + 1].mean()), 2) for a0, b0 in wins]),
               per_frame_deg={int(f): round(float(corr[f]), 2) for f in applied})
    (out / 'tiltfix_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_frame_deg'}))


if __name__ == '__main__':
    main()
