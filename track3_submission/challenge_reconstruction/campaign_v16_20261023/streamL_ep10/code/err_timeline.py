"""Per-frame official-style (initial-scene-aligned) cup errors of trajectories vs our reference, binned in 0.5 s windows.
usage: err_timeline.py EPISODE_DIR TRAJ [TRAJ...]"""
import sys, json, numpy as np
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import load_reference, load_trajectory, resample_to_reference, apply_world_transform, validate_geometry, align_initial_scene, rotation_matrices, angular_errors, validate_poses
ep = Path(sys.argv[1]); man = json.loads((ep / 'input_manifest.json').read_text()); geom = json.loads((ep / 'metric_geometry.json').read_text())
ref = validate_poses(load_reference(man['source'])); anchors, axes, kinds = validate_geometry(ref['object_names'], geom)
ref['object_positions'] = ref['object_positions'] + np.einsum('twoij,oj->twoi', rotation_matrices(ref['object_quaternions_wxyz']), anchors)
t = ref['timestamps_seconds']; edges = np.arange(0, t[-1] + 0.5, 0.5)
for p in sys.argv[2:]:
    tr = validate_poses(load_trajectory(p)); tr = resample_to_reference(tr, ref); tr = apply_world_transform(tr, np.asarray(man['source_from_setup']))
    tr['object_positions'] = tr['object_positions'] + np.einsum('twoij,oj->twoi', rotation_matrices(tr['object_quaternions_wxyz']), anchors)
    reg, _ = align_initial_scene(tr, ref, axes, kinds)
    pos = np.linalg.norm(reg['object_positions'] - ref['object_positions'], axis=-1)[:, 0] * 100; ang = np.rad2deg(angular_errors(reg['object_quaternions_wxyz'], ref['object_quaternions_wxyz'], axes, kinds))[:, 0]
    dz = (reg['object_positions'] - ref['object_positions'])[:, 0, :, 2] * 100
    print('##', p[-90:], ' mean:', {n: (round(pos[:, i].mean(), 2), round(ang[:, i].mean(), 1)) for i, n in enumerate(ref['object_names'])})
    print('   window   | ' + ' | '.join(f'{n[:9]:>9} cm/deg (dz)' for n in ref['object_names']))
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (t >= lo) & (t < hi + 1e-9)
        if not s.any(): continue
        print(f'  {lo:4.1f}-{hi:4.1f} s | ' + ' | '.join(f'{pos[s, i].mean():5.2f}/{ang[s, i].mean():5.1f} ({dz[s, i].mean():+5.2f})' for i in range(len(ref['object_names']))))
