"""spoon3537: per-object time-mean and per-0.5 s window errors (cm / deg, kit MP-SR registered alignment) of a build-v2 candidate's
results parquet, robot (achieved) vs its own reference columns (= the target it was built against), using the FINAL component's metric
geometry (copy of the review scorer, episode_review/code/review_ep.py). usage: s3win.py EP COMPONENT_DIR CAND_EPISODE_DIR [CAND2_EPISODE_DIR ...]"""
import sys, os, json, importlib.util
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, trimesh
EP = int(sys.argv[1]); COMP = Path(sys.argv[2]); E = f'episode_{EP:06d}'
KIT = Path('/data1/SRL_V2D/challenge_submissions_SRL/FINAL_20261104/all_20_episodes/submission_kit/metric_code/track_3/MP-SR.py')
spec = importlib.util.spec_from_file_location('mpsr', KIT); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
PV = COMP / 'portable_validation'
man = json.loads((PV / f'runtime/episodes/{E}/input_manifest.json').read_text()); geo = json.loads((PV / f'runtime/episodes/{E}/metric_geometry.json').read_text())
src = Path(man['source']); op = np.load(src / 'object_poses.npz'); names = [str(n) for n in op['names']]; T = int(man['frames']); fps = float(man['fps'])
axes, orders, anchors = [], [], []
for n in names:
    g = geo['objects'][n]; ax = np.array(g.get('axis_local', [0, 0, 0]), float); sym = str(g.get('symmetry', 'none'))
    orders.append(int(g.get('order', 2 if '2' in sym or 'mirror' in sym else (360 if np.any(ax) else 1)))); axes.append(ax); anchors.append(np.array(g['anchor_local_m'], float))


def load(d):
    tab = pq.read_table(Path(d) / f'results/{E}.parquet').to_pandas(); tab = tab[tab.environment_id == tab.environment_id.min()]
    def arr(prefix):
        a = np.zeros((T, 1, len(names), 7))
        for b, n in enumerate(names):
            g = tab[tab.object_body_name == n].sort_values('step')
            a[:, 0, b, :3] = g[[f'{prefix}_position_{c}' for c in 'xyz']].values; a[:, 0, b, 3:] = g[[f'{prefix}_quaternion_{c}' for c in 'xyzw']].values
        a[..., 3:] /= np.linalg.norm(a[..., 3:], axis=-1, keepdims=True)
        for i in range(len(names)):
            a[:, :, i, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(a[:, :, i, 3:]), anchors[i])
        return a
    ach, ref = arr('achieved'), arr('reference')
    if os.environ.get('TGTDIR'):
        from scipy.spatial.transform import Rotation as Rr
        for b, n in enumerate(names):
            Tt = np.load(Path(os.environ['TGTDIR']) / f'{n}_trajectory.npz')['T_world_obj'][:T]
            ref[:, 0, b, :3] = Tt[:, :3, 3] + np.einsum('tij,j->ti', Tt[:, :3, :3], anchors[b]); ref[:, 0, b, 3:] = Rr.from_matrix(Tt[:, :3, :3]).as_quat()
    al = M._align_registered_scene(ach, ref, axes)
    pe = np.linalg.norm(al[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100; re = np.degrees(M._symmetry_orientation_errors(al, ref, axes, orders))[:, 0]
    return pe, re


W = int(round(0.5 * fps)); WR = [float(x) for x in os.environ.get('WR', '5.0,9.5').split(',')]
for d in sys.argv[3:]:
    pe, re = load(d); print('==', d)
    for i, n in enumerate(names):
        wins = [(k / fps, float(pe[k:k + W, i].mean()), float(re[k:k + W, i].mean())) for k in range(0, T, W)]
        wp = max(wins, key=lambda x: x[1]); wr = max(wins, key=lambda x: x[2])
        print(f'  {n:14s} mean {pe[:, i].mean():.2f} cm / {re[:, i].mean():.2f} deg | worst pos win {wp[0]:.1f}s {wp[1]:.2f}cm/{wp[2]:.1f}deg | worst rot win {wr[0]:.1f}s {wr[1]:.2f}cm/{wr[2]:.1f}deg')
        print(f'   0.5s windows {WR[0]}-{WR[1]} s (cm/deg):', ' '.join(f'{t:.1f}:{p:.1f}/{r:.0f}' for t, p, r in wins if WR[0] <= t < WR[1]))
