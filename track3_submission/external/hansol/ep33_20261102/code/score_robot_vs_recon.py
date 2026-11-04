#!/usr/bin/env python3
"""Robot object poses (pir_objects_to_world.py output) vs a reconstruction target, the kit's way (runs on the lab server):
same meshes on both sides, so registration is the identity; anchors = mesh surface centroids; scene aligned at frame 0
by M._align_registered_scene (slot-0 position, first asymmetric object's rotation); symmetry-aware orientation error.
Robot poses on the RGB clock are first moved to the stored clock (+delta). Reports per object mean cm / deg, the share of
frames over 1.5 cm / 15 deg and 3 cm / 30 deg, and window means; GT-free.
usage: score_robot_vs_recon.py ROBOT_NPZ TARGET_EPISODE_DIR EPISODE OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
sys.path.insert(0, '/data1/hansol/mink_S1_20261031/code')
import public_kit_score as K  # noqa: E402
from contact_smooth import DELTA, shift  # noqa: E402


def main():
    rob, tgt, ep, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    M = K.load_metric('MP-SR'); z = np.load(rob); names = [str(n) for n in z['names']]
    Wr = z['world_from_object'].astype(float); T = len(Wr)
    if str(z['clock']) == 'rgb':
        d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:T] if np.ndim(d) else np.full(T, float(d))
        Wr = np.stack([shift(Wr[:, k], d) for k in range(len(names))], 1)
    order = sorted(names, key=lambda n: ['blue_cup', 'wooden_spoon'].index(n) if n in ('blue_cup', 'wooden_spoon') else 9)
    Wt = np.stack([np.load(tgt / f'{n}_trajectory.npz')['T_world_obj'][:T] for n in order], 1)
    Wr = np.stack([Wr[:, names.index(n)] for n in order], 1)
    def tens(W):
        X = np.zeros((T, 1, len(order), 7))
        for k in range(len(order)):
            c = K.pose_cols(W[:, k]); X[:, 0, k] = np.c_[c['pos_x'], c['pos_y'], c['pos_z'], c['quat_x'], c['quat_y'], c['quat_z'], c['quat_w']]  # kit order: xyzw
        return X
    ach, ref = tens(Wr), tens(Wt); axes, orders = [], []
    for k, n in enumerate(order):
        m = trimesh.load(tgt / f'{n}_mesh_metric.obj', force='mesh'); v = np.asarray(m.vertices, float); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(n, v); axes.append(ax); orders.append(int(o))
        for X in (ach, ref):
            X[:, :, k, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(X[:, :, k, 3:]), a)
    ach = M._align_registered_scene(ach, ref, axes)
    pos = np.linalg.norm(ach[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(ach, ref, axes, orders))[:, 0]
    res = {}
    for k, n in enumerate(order):
        p, o = pos[:, k], ori[:, k]
        res[n] = dict(mean_cm=round(float(p.mean()), 2), mean_deg=round(float(o.mean()), 1), pass_1p5_15=bool(p.mean() < 1.5 and o.mean() < 15),
                      pass_3_30=bool(p.mean() < 3 and o.mean() < 30), frac_over_1p5cm=round(float((p > 1.5).mean()), 3), frac_over_15deg=round(float((o > 15).mean()), 3),
                      windows={f'f{a}-{min(b, T) - 1}': [round(float(p[a:b].mean()), 2), round(float(o[a:b].mean()), 1)] for a, b in ((0, 53), (53, 124), (124, 181), (181, 260), (260, T)) if a < T},
                      cm=[round(float(x), 2) for x in p], deg=[round(float(x), 1) for x in o])
    out.write_text(json.dumps(dict(robot=str(rob), target=str(tgt), objects=res), indent=1))
    print(json.dumps({n: {k: v for k, v in r.items() if k not in ('cm', 'deg')} for n, r in res.items()}, indent=1))


if __name__ == '__main__':
    main()
