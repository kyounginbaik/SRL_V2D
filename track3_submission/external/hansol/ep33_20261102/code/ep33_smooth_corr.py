#!/usr/bin/env python3
"""Smooth a per-frame spoon correction over time (GT-free). The roll refit (spoon_annot_roll.py) works on a 10 deg grid,
so its correction comes in steps. Here the body-frame correction B_k = inv(base_k) @ new_k of the spoon's
world_from_object is turned into (rotation vector, translation), Gaussian-filtered over frames (--sigma, stored clock) and
re-applied to the base as base_k @ B_smooth_k for world_from_object and camera_from_object. Other objects are untouched.
usage: ep33_smooth_corr.py BASE_EPISODE_DIR NEW_EPISODE_DIR OUT_EPISODE_DIR [--sigma 1.5]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot


def main():
    base, new, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sig = float(sys.argv[sys.argv.index('--sigma') + 1]) if '--sigma' in sys.argv else 1.5
    zb, zn = dict(np.load(base / 'object_poses.npz')), dict(np.load(new / 'object_poses.npz'))
    s = [str(n) for n in zb['names']].index('wooden_spoon')
    Wb, Wn = zb['world_from_object'][:, s].astype(float), zn['world_from_object'][:, s].astype(float)
    B = np.stack([np.linalg.inv(a) @ b for a, b in zip(Wb, Wn)])
    rv, tv = Rot.from_matrix(B[:, :3, :3]).as_rotvec(), B[:, :3, 3]
    rvs, tvs = gaussian_filter1d(rv, sig, axis=0, mode='nearest'), gaussian_filter1d(tv, sig, axis=0, mode='nearest')
    wfo, cfo = zn['world_from_object'].copy(), zn['camera_from_object'].copy()
    for k in range(len(B)):
        if np.linalg.norm(rvs[k]) < 1e-6 and np.linalg.norm(tvs[k]) < 1e-7:  # untouched frames (frame 0 above all) stay bit-exact
            wfo[k, s], cfo[k, s] = zb['world_from_object'][k, s], zb['camera_from_object'][k, s]
            continue
        Bs = np.eye(4); Bs[:3, :3] = Rot.from_rotvec(rvs[k]).as_matrix(); Bs[:3, 3] = tvs[k]
        wfo[k, s] = (zb['world_from_object'][k, s].astype(float) @ Bs).astype(wfo.dtype)
        cfo[k, s] = (zb['camera_from_object'][k, s].astype(float) @ Bs).astype(cfo.dtype)
    zn['world_from_object'], zn['camera_from_object'] = wfo, cfo
    if out.resolve() != new.resolve():
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(new, out)
    np.savez_compressed(out / 'object_poses.npz', **zn)
    step = np.degrees(np.linalg.norm(np.diff(rv, axis=0), axis=1)); steps = np.degrees(np.linalg.norm(np.diff(rvs, axis=0), axis=1))
    rep = dict(sigma=sig, correction_deg_max=[round(float(np.degrees(np.linalg.norm(rv, axis=1)).max()), 1), round(float(np.degrees(np.linalg.norm(rvs, axis=1)).max()), 1)],
               correction_step_deg_max=[round(float(step.max()), 1), round(float(steps.max()), 1)])
    (out / 'smooth_corr_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == '__main__':
    main()
