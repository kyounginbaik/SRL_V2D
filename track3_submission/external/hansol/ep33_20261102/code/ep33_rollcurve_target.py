#!/usr/bin/env python3
"""ep33 spoon roll correction along a knot curve (GT-free), written as a score_vs_compose target and optionally as a bundle
episode. Same convention as ep33_restroll_target.py / K33 imgev: roll about the mesh's long PCA axis through its centre of
mass, applied in the body frame (world_from_object @ B). Between knots the correction follows a smoothstep; before the
first knot it holds the first value, after the last knot the last value.
usage: ep33_rollcurve_target.py BUNDLE_EPISODE_DIR OUT_TARGET_EPISODE_DIR "t0:c0,t1:c1,..." [--bundle-out DIR]
example: "0:-8,22:-8,34:4,52:4,64:0" """
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot


def curve(knots, n):
    t = np.array([k[0] for k in knots], float); c = np.array([k[1] for k in knots], float)
    sm = lambda u: u ** 3 * (10 - 15 * u + 6 * u ** 2)
    out = np.empty(n)
    for f in range(n):
        if f <= t[0]:
            out[f] = c[0]
        elif f >= t[-1]:
            out[f] = c[-1]
        else:
            i = int(np.searchsorted(t, f, side='right') - 1); u = (f - t[i]) / (t[i + 1] - t[i])
            out[f] = c[i] + (c[i + 1] - c[i]) * sm(u)
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    knots = [tuple(map(float, k.split(':'))) for k in sys.argv[3].split(',')]
    z = np.load(src / 'object_poses.npz'); d = {k: z[k] for k in z.files}; names = [str(n) for n in d['names']]; KS = names.index('wooden_spoon')
    mj = json.loads((src / 'meshes.json').read_text()); mfile = lambda n: src / (mj[n]['file'] if isinstance(mj[n], dict) else mj[n])
    m = trimesh.load(mfile('wooden_spoon'), force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
    c0 = v - v.mean(0); _, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
    if long[np.argmax(np.abs(long))] < 0:
        long = -long
    n = len(d['world_from_object']); corr = curve(knots, n)

    def apply(A):
        A = A.astype(float).copy()
        for f in np.flatnonzero(corr != 0):
            E = Rot.from_rotvec(long * np.radians(corr[f])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
            A[f, KS] = A[f, KS] @ B
        return A
    W = apply(d['world_from_object'])
    out.mkdir(parents=True, exist_ok=True)
    for k, nm in enumerate(names):
        np.savez_compressed(out / f'{nm}_trajectory.npz', T_world_obj=W[:, k])
        trimesh.load(mfile(nm), force='mesh', process=False).export(out / f'{nm}_mesh_metric.obj')
    rec = dict(source=str(src), rule='roll knot curve (smoothstep between knots), long PCA axis through COM', knots=knots,
               correction_deg={str(f): round(float(corr[f]), 2) for f in range(0, min(n, 80), 2)}, frames_changed=int((corr != 0).sum()),
               ground_truth_used=False)
    (out / 'ROLLCURVE.json').write_text(json.dumps(rec, indent=1))
    if '--bundle-out' in sys.argv:
        bo = Path(sys.argv[sys.argv.index('--bundle-out') + 1])
        if bo.exists():
            shutil.rmtree(bo)
        shutil.copytree(src, bo)
        d2 = dict(d); d2['world_from_object'] = W.astype(d['world_from_object'].dtype)
        d2['camera_from_object'] = apply(d['camera_from_object']).astype(d['camera_from_object'].dtype)
        np.savez_compressed(bo / 'object_poses.npz', **d2); (bo / 'ROLLCURVE.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ('knots', 'frames_changed')}))


if __name__ == '__main__':
    main()
