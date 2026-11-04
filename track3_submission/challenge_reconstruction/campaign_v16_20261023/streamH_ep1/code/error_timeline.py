"""Per-frame official-style (frame-0 aligned) and declared-frame errors of a rollout trajectory against one or more
reference dirs. usage: error_timeline.py TRAJ.npz MANIFEST.json GEOMETRY.json OUT.json REF_DIR [REF_DIR ...]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code')
import evaluate_mp_sr as M
traj, manifest, geometry, out = sys.argv[1:5]; refs = sys.argv[5:]
man = json.loads(Path(manifest).read_text()); geo = json.loads(Path(geometry).read_text())
T = M.apply_world_transform(M.load_trajectory(traj), np.asarray(man['source_from_setup'], float))
report = {}
for rd in refs:
    ref = M.load_reference(rd); tr = M.resample_to_reference(dict(T), ref)
    anchors, axes, kinds = M.validate_geometry(ref['object_names'], geo)
    ref = dict(ref); tr = dict(tr)
    for d in (ref, tr):
        d['object_positions'] = d['object_positions'] + np.einsum('twoij,oj->twoi', M.rotation_matrices(d['object_quaternions_wxyz']), anchors)
    reg, tfs = M.align_initial_scene(tr, ref, axes, kinds)
    rows = {}
    for tag, a in (('declared', tr), ('official', reg)):
        pos = np.linalg.norm(a['object_positions'] - ref['object_positions'], axis=-1)[:, 0] * 100
        ang = np.rad2deg(M.angular_errors(a['object_quaternions_wxyz'], ref['object_quaternions_wxyz'], axes, kinds))[:, 0]
        rows[tag] = {n: dict(pos_cm=pos[:, i].round(3).tolist(), deg=ang[:, i].round(2).tolist(), mean_cm=float(pos[:, i].mean()), mean_deg=float(ang[:, i].mean()), max_cm=float(pos[:, i].max()), max_deg=float(ang[:, i].max())) for i, n in enumerate(ref['object_names'])}
    from scipy.spatial.transform import Rotation as R
    rows['frame0_alignment_rotation_deg'] = float(np.degrees(R.from_matrix(np.asarray(tfs[0])[:3, :3]).magnitude()))
    report[rd] = rows
    t = ref['timestamps_seconds']
    print('==', rd, 'frame0 align rot %.1f deg' % rows['frame0_alignment_rotation_deg'])
    for tag in ('declared', 'official'):
        print(' ', tag, {n: (round(v['mean_cm'], 2), round(v['mean_deg'], 1), 'max', round(v['max_cm'], 1)) for n, v in rows[tag].items()})
    # 1-s bins of official error
    n = len(t); edges = np.arange(0, t[-1] + 1, 1.0)
    for name in ref['object_names']:
        p = np.asarray(rows['official'][name]['pos_cm']); d = np.asarray(rows['official'][name]['deg'])
        line = ' '.join('%4.1f/%4.1f' % (p[(t >= e) & (t < e + 1)].mean(), d[(t >= e) & (t < e + 1)].mean()) for e in edges if ((t >= e) & (t < e + 1)).any())
        print('  %-14s 1s bins cm/deg: %s' % (name, line))
Path(out).write_text(json.dumps(report, indent=0))
