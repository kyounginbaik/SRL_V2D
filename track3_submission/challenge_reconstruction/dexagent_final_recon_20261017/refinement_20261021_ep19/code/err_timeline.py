"""Per-frame official-style errors (anchor offset + one initial-scene alignment, as evaluate_mp_sr) printed per second.
usage: err_timeline.py TRAJ.npz INPUT_MANIFEST.json METRIC_GEOMETRY.json [--npz OUT]"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code')
import evaluate_mp_sr as E
def timeline(traj, manifest, geometry):
    man = json.loads(Path(manifest).read_text()); geo = json.loads(Path(geometry).read_text())
    ref = E.validate_poses(E.load_reference(Path(man['source']))); tr = E.validate_poses(E.load_trajectory(traj))
    tr = E.resample_to_reference(tr, ref); tr = E.apply_world_transform(tr, np.asarray(man['source_from_setup'], float))
    anchors, axes, kinds = E.validate_geometry(ref['object_names'], geo)
    for d in (ref, tr): d['object_positions'] += np.einsum('twoij,oj->twoi', E.rotation_matrices(d['object_quaternions_wxyz']), anchors)
    reg, _ = E.align_initial_scene(tr, ref, axes, kinds)
    pos = np.linalg.norm(reg['object_positions'] - ref['object_positions'], axis=-1)[:, 0] * 100
    ang = np.degrees(E.angular_errors(reg['object_quaternions_wxyz'], ref['object_quaternions_wxyz'], axes, kinds))[:, 0]
    return list(ref['object_names']), pos, ang, ref['timestamps_seconds']
if __name__ == '__main__':
    names, pos, ang, t = timeline(*sys.argv[1:4]); n = len(t)
    for b, nm in enumerate(names):
        secs = range(0, int(t[-1]) + 1); idx = lambda s: min(int(np.searchsorted(t, s)), n - 1)
        print(f'{nm:14s} cm :', ' '.join(f'{s}:{pos[idx(s), b]:.1f}' for s in secs), f'| mean {pos[:, b].mean():.2f}')
        print(f'{"":14s} deg:', ' '.join(f'{s}:{ang[idx(s), b]:.0f}' for s in secs), f'| mean {ang[:, b].mean():.1f}')
    if '--npz' in sys.argv: np.savez(sys.argv[sys.argv.index('--npz') + 1], names=names, pos=pos, ang=ang, t=t)
