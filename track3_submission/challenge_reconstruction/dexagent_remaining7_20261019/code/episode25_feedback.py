#!/usr/bin/env python3
"""Measured object-pose feedback for episode 25: correct each hand's wrist commands
by the error its board showed in the previous physical rollout, then re-execute.

Generalisation of cup20_feedback.py: one correction per (hand, object) pair,
applied only while that hand actually touched that object in the parent rollout
(dilated by a short blend). Object targets, timestamps, finger commands, scene
and the start state stay fixed; only robot wrist goals move. Every output is a
fresh full-episode physics execution with the unchanged proxy evaluator.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, json, os, sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from prepare_seven import BASE, LEGACY, ROOT, write
sys.path.insert(0, str(LEGACY / 'code')); sys.path.insert(0, str(ROOT / 'third_party/DexAgent'))
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import blend
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files


def contact_mask(m, actual, side, obj):
    """Frames where this hand touched this object, recomputed from the parent rollout states."""
    body = m.body(obj).id
    robot = {b for b in range(m.nbody) if (m.body(b).name or '').startswith(f'robot_{side}_')}
    d = mujoco.MjData(m); mask = np.zeros(len(actual), bool)
    for t in range(len(actual)):
        d.qpos[:] = actual[t]; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == body and b2 in robot) or (b2 == body and b1 in robot):
                mask[t] = True; break
    return mask


def trial(task):
    source, inp, out, gain, pairs, blend_s, max_deg = task
    source, inp, out = Path(source), Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((source / 'result.json').read_text())
    manifest = json.loads((inp / 'input_manifest.json').read_text())
    scene = old['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
    ref = TR.load_reference(old['reference'], m); Q = ref['qpos'].copy(); saved = Q.copy()
    fps = float(ref['fps']); times = np.arange(len(Q)) / fps
    actual = np.load(old['execution']['trajectory'])['qpos']
    T = min(len(Q), len(actual)); assert T == len(Q), (T, len(Q), len(actual))
    geometry = json.loads((inp / 'metric_geometry.json').read_text())['objects']
    touched_cols = []
    applied = {}
    for side, obj in pairs:
        rig = H.HandRig.build(m, f'robot_{side}_', side)
        wc = np.array([rig.wrist_qpos[j] for j in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
        fc = np.array(list(rig.finger_qpos.values())); cols = np.r_[wc, fc]; touched_cols.append(cols)
        a = int(m.joint(obj + '_free').qposadr[0])
        R = Rotation.from_quat(Q[:, a + 3:a + 7], scalar_first=True)
        S = Rotation.from_quat(actual[:, a + 3:a + 7], scalar_first=True)
        axis = np.array(geometry.get(obj, {}).get('axis_local', [0, 0, 0]), float)
        if np.linalg.norm(axis) > 1e-8:
            axis /= np.linalg.norm(axis)
            vr = R.apply(np.tile(axis, (len(Q), 1))); vs = S.apply(np.tile(axis, (len(Q), 1)))
            cross = np.cross(vs, vr); angles = np.arctan2(np.linalg.norm(cross, axis=1), np.sum(vs * vr, axis=1))
            rv = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-9) * np.minimum(angles, np.radians(max_deg))[:, None]
        else:
            rv = (R * S.inv()).as_rotvec()
            rv *= np.minimum(1., np.radians(max_deg) / np.maximum(np.linalg.norm(rv, axis=1), 1e-9))[:, None]
        delta = Q[:, a:a + 3] - actual[:, a:a + 3]
        hold = contact_mask(m, actual, side, obj)
        if not hold.any():
            applied[f'{side}:{obj}'] = dict(contact_frames=0, skipped=True); continue
        # Smooth weight: 1 while holding, blended in/out over blend_s seconds.
        w = hold.astype(float)
        k = max(1, int(round(blend_s * fps)))
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        w = np.clip(np.convolve(w, kernel, mode='same') * (2 * k + 1) / (k + 1), 0, 1)
        w = savgol_filter(w, 2 * k + 1 if (2 * k + 1) % 2 else 2 * k + 3, 2)
        w = np.clip(w, 0, 1)
        delta = savgol_filter(delta, 9, 2, axis=0); rv = savgol_filter(rv, 9, 2, axis=0)
        rv = rv * gain * w[:, None]; delta = delta * gain * w[:, None]
        D = Rotation.from_rotvec(rv); center = Q[:, a:a + 3]
        Q[:, wc[:3]] = center + D.apply(Q[:, wc[:3]] - center) + delta
        Q[:, wc[3:]] = continuous_euler_xyz(D * Rotation.from_euler('XYZ', Q[:, wc[3:]]), Q[0, wc[3:]])
        applied[f'{side}:{obj}'] = dict(contact_frames=int(hold.sum()),
            first_contact_s=float(times[hold][0]), last_contact_s=float(times[hold][-1]),
            max_position_shift_m=float(np.linalg.norm(delta, axis=1).max()),
            max_rotation_shift_deg=float(np.degrees(np.linalg.norm(rv, axis=1).max())))
    Q[0] = saved[0]
    other = np.setdiff1d(np.arange(m.nq), np.concatenate(touched_cols))
    assert np.array_equal(Q[:, other], saved[:, other]); assert np.isfinite(Q).all()
    robot = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    Q[:, robot] = np.clip(Q[:, robot], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    path = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], fps, ref['segment'], ref['segments'],
        source='Robot-only wrist correction from measured board pose feedback; unchanged object targets', floating_runtime=True)
    execution = execute_reference(scene, path, out / 'execution', expected_duration_s=(len(Q) - 1) / fps)
    metrics = evaluate_files(manifest['source'], execution['trajectory'], inp / 'input_manifest.json',
                             execution['scene'], inp / 'metric_geometry.json')
    result = dict(episode=inp.name, input_dir=str(inp), reference=str(path), execution=execution, metrics=metrics,
        parent=str(source), gain=gain, pairs=[list(p) for p in pairs], blend_s=blend_s, max_deg=max_deg,
        applied=applied, object_targets_unchanged=True, learned_policy=False, submission_ready=False)
    write(out / 'result.json', result)
    print(json.dumps(dict(output=str(out), mp_sr=metrics['reference_proxy_mp_sr'],
        objects={k: dict(cm=v['mean_position_error_cm'][0], deg=v['mean_orientation_error_degrees'][0])
                 for k, v in metrics['official_style_reference_proxy']['objects'].items()})), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--episode-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pair', action='append', default=[], help='side=object, repeatable')
    p.add_argument('--gains', type=float, nargs='+', default=[.5, 1.])
    p.add_argument('--blend', type=float, nargs='+', default=[.3])
    p.add_argument('--max-deg', type=float, default=28.)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    pairs = [tuple(v.split('=')) for v in (args.pair or ['left=wooden_piece_1', 'right=wooden_piece_2'])]
    tasks = [(str(args.source), str(args.episode_dir), str(args.output / f'gain{g:g}_blend{b:g}'), g, pairs, b, args.max_deg)
             for g in args.gains for b in args.blend]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(trial, tasks))
    write(args.output / 'summary.json', [dict(reference=r['reference'], metrics=r['metrics'], applied=r['applied']) for r in results])
