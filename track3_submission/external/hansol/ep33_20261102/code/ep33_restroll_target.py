#!/usr/bin/env python3
"""K33's frame-0 rest-roll rule (streamK33_ep33/code/build_h33.py) applied to any ep33 bundle, written as a
score_vs_compose target (GT-free). The spoon is rolled about its long PCA axis through its centre of mass by c on f0..t_s,
ramped back with a smoothstep until t_e; c = 'rest' is the median over f0..t_s of the per-frame roll at which our rigid
spoon mesh lies lowest on our stereo support plane (roll scan +-60 deg, 1 deg). The cup comes from the same bundle.
usage: ep33_restroll_target.py BUNDLE_EPISODE_DIR OUT_TARGET_EPISODE_DIR c|rest t_s t_e"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

CAN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033')


def main():
    src, out, cs, ts, te = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    z = np.load(src / 'object_poses.npz'); d = {k: z[k] for k in z.files}; names = [str(n) for n in d['names']]; KS = names.index('wooden_spoon')
    mj = json.loads((src / 'meshes.json').read_text())
    mfile = lambda n: src / (mj[n]['file'] if isinstance(mj[n], dict) else mj[n])
    m = trimesh.load(mfile('wooden_spoon'), force='mesh'); v = np.asarray(m.vertices); com = np.asarray(m.center_mass)
    c0 = v - v.mean(0); _, U = np.linalg.eigh(c0.T @ c0 / len(c0)); long = U[:, 2]
    if long[np.argmax(np.abs(long))] < 0:
        long = -long
    G = np.load(CAN / 'support_plane.npz')['gravity_from_canonical']

    def rest_roll(t):
        P = G @ d['world_from_object'][t, KS]; cg = P[:3, :3] @ com + P[:3, 3]; best = None
        for r in np.arange(-60, 61, 1.0):
            E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); vg = ((v - com) @ E.T + com) @ P[:3, :3].T + P[:3, 3]
            h = cg[2] - vg[:, 2].min(); best = min(best or (h, r), (h, r))
        return best[1]
    per = [rest_roll(t) for t in range(ts + 1)] if cs == 'rest' else []
    c = float(np.median(per)) if cs == 'rest' else float(cs)
    sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
    corr = np.array([c if t <= ts else (c * (1 - sm((t - ts) / (te - ts))) if t < te else 0.0) for t in range(len(d['world_from_object']))])
    W = d['world_from_object'].astype(float).copy()
    for t in np.flatnonzero(corr != 0):
        E = Rot.from_rotvec(long * np.radians(corr[t])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
        W[t, KS] = W[t, KS] @ B
    out.mkdir(parents=True, exist_ok=True)
    for k, n in enumerate(names):
        np.savez_compressed(out / f'{n}_trajectory.npz', T_world_obj=W[:, k])
        trimesh.load(mfile(n), force='mesh', process=False).export(out / f'{n}_mesh_metric.obj')
    rec = dict(source=str(src), rule='K33 build_h33 rest roll', c_deg=c, c_mode=cs, per_frame_rest_roll_deg=per, t_s=ts, t_e=te,
               frames_changed=int((corr != 0).sum()), ground_truth_used=False)
    if '--bundle-out' in sys.argv:  # the same poses as a bundle episode (copy of the source with object_poses replaced)
        import shutil
        bo = Path(sys.argv[sys.argv.index('--bundle-out') + 1])
        if bo.exists():
            shutil.rmtree(bo)
        shutil.copytree(src, bo)
        d2 = dict(d); d2['world_from_object'] = W.astype(d['world_from_object'].dtype)
        C = d['camera_from_object'].astype(float).copy()
        for t in np.flatnonzero(corr != 0):
            E = Rot.from_rotvec(long * np.radians(corr[t])).as_matrix(); B = np.eye(4); B[:3, :3] = E; B[:3, 3] = com - E @ com
            C[t, KS] = C[t, KS] @ B
        d2['camera_from_object'] = C.astype(d['camera_from_object'].dtype)
        np.savez_compressed(bo / 'object_poses.npz', **d2); (bo / 'RESTROLL.json').write_text(json.dumps(rec, indent=1))
    (out / 'RESTROLL.json').write_text(json.dumps(rec, indent=1)); print(json.dumps({k: rec[k] for k in ('source', 'c_deg', 'frames_changed')}))


if __name__ == '__main__':
    main()
