#!/usr/bin/env python3
"""Check CEM search/replay consistency, separately from manipulation success.

This checks a completed research run. Passing is not a challenge submission
certificate and does not turn MPC outputs into a learned-policy checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from evaluate_mp_sr import evaluate_files

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_run(directory, position_tolerance_cm=1e-5, angle_tolerance_deg=1e-4):
    directory = Path(directory).resolve()
    problems = []
    result, settings = read(directory/'result.json'), read(directory/'settings.json')
    spec, execution = settings['spec'], result['execution']
    source = B/'episodes'/result['episode']
    model_search = mujoco.MjModel.from_xml_path(spec['scene'])
    model_final = mujoco.MjModel.from_xml_path(execution['scene'])
    changed_arrays = []
    for key in dir(model_search):
        if not key.startswith(('body_', 'jnt_', 'geom_', 'actuator_', 'pair_', 'exclude_', 'dof_', 'mesh_', 'qpos0', 'key_')):
            continue
        a, b = getattr(model_search, key), getattr(model_final, key)
        if isinstance(a, np.ndarray) and not np.array_equal(a, b):
            changed_arrays.append(key)
    changed_options = []
    for key in dir(model_search.opt):
        if key.startswith('_'):
            continue
        a, b = getattr(model_search.opt, key), getattr(model_final.opt, key)
        if not callable(a) and not np.array_equal(a, b):
            changed_options.append(key)
    if changed_arrays or changed_options:
        problems.append('Search/final model physics differ')
    for key, expected in settings.get('source_sha256', {}).items():
        if key not in spec or sha(spec[key]) != expected:
            problems.append(f'Search input hash mismatch: {key}')
    with np.load(execution['trajectory'], allow_pickle=False) as z:
        measured_start = z['initial_qpos']
        names = z['object_names'].astype(str).tolist()
        recording_scene_hash = str(z['scene_sha256'])
        measured_frames = len(z['qpos'])
    if recording_scene_hash != sha(execution['scene']):
        problems.append('Recording scene hash mismatch')
    home = model_search.key_qpos[0] if model_search.nkey else model_search.qpos0
    initial_difference = float(np.max(np.abs(measured_start-home)))
    if not np.array_equal(measured_start, home):
        problems.append('Final execution did not start at exact search scene home')
    with np.load(result['reference'], allow_pickle=False) as selected, np.load(spec['original_reference'], allow_pickle=False) as original:
        selected_names = selected['joint_names'].astype(str).tolist()
        original_names = original['joint_names'].astype(str).tolist()
        if len(set(selected_names)) != len(selected_names) or set(selected_names) != set(original_names):
            raise ValueError('Reference joint names are inconsistent')
        order = [selected_names.index(name) for name in original_names]
        q = selected['qpos'][:, order]
        raw = original['qpos']
        if q.shape != raw.shape or float(selected['fps']) != float(original['fps']) or measured_frames != len(raw):
            problems.append('Reference/recording full horizon mismatch')
        object_columns = [i for i, name in enumerate(original_names) if not name.startswith('robot_')]
        if not np.array_equal(q[:, object_columns], raw[:, object_columns]):
            problems.append('Refinement changed immutable object targets')
    manifest = read(source/'input_manifest.json')
    metrics = evaluate_files(manifest['source'], execution['trajectory'], source/'input_manifest.json',
                             execution['scene'], source/'metric_geometry.json')
    scored = metrics['declared_frame_reference_proxy']['objects']
    final_pos = np.array([scored[o]['mean_position_error_cm'][0] for o in names])
    final_ang = np.array([scored[o]['mean_orientation_error_degrees'][0] for o in names])
    search_pos = np.array(result['best']['mean_position_cm'])
    search_ang = np.array(result['best']['mean_orientation_deg'])
    dp, da = np.abs(final_pos-search_pos), np.abs(final_ang-search_ang)
    if not np.isfinite(dp).all() or np.max(dp) > position_tolerance_cm:
        problems.append('Search/final mean position errors disagree beyond tolerance')
    if not np.isfinite(da).all() or np.max(da) > angle_tolerance_deg:
        problems.append('Search/final mean orientation errors disagree beyond tolerance')
    if float(result['best']['mp_sr']) != metrics['declared_frame_mp_sr']:
        problems.append('Search/final success flags disagree')
    if not execution.get('physics_verified') or not execution.get('execution_completed'):
        problems.append('Final physics validation/full horizon failed')
    replay = execution.get('independent_control_replay', {})
    if not replay.get('ok'):
        problems.append('Missing passing independent control replay')
    return dict(episode=result['episode'], experiment=str(directory), consistent=not problems,
                problems=problems, changed_model_arrays=changed_arrays, changed_model_options=changed_options,
                exact_scene_home_difference=initial_difference, object_names=names,
                search_mean_position_cm=search_pos.tolist(), final_mean_position_cm=final_pos.tolist(),
                search_mean_orientation_deg=search_ang.tolist(), final_mean_orientation_deg=final_ang.tolist(),
                max_position_disagreement_cm=float(dp.max()), max_angle_disagreement_deg=float(da.max()),
                position_tolerance_cm=position_tolerance_cm, angle_tolerance_deg=angle_tolerance_deg,
                task_reference_proxy_mp_sr=metrics['reference_proxy_mp_sr'],
                physics_verified=execution['physics_verified'], full_horizon=execution['execution_completed'],
                learned_policy=False, official_submission_ready=False,
                note='Consistency is independent of manipulation success. Reconstructed reference is proxy GT; contact/grasp audit is separate.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--position-tolerance-cm', type=float, default=1e-5)
    p.add_argument('--angle-tolerance-deg', type=float, default=1e-4)
    args = p.parse_args()
    report = check_run(args.run, args.position_tolerance_cm, args.angle_tolerance_deg)
    output = args.output or args.run/'audit/search_replay_consistency.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['consistent'] else 2)


if __name__ == '__main__':
    main()
