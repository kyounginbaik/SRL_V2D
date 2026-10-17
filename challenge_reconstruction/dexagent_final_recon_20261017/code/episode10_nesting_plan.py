#!/usr/bin/env python3
"""Object-only cup fit/settling PLANNING; deliberately not an episode evaluation.

Initial poses are changed and robot bodies removed in a separate derived scene.
All cup geometry, mass, friction, solver settings and the finite table are retained.
Never feed this recording to an episode scorer or promote it as task success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

B = Path(__file__).resolve().parents[1]
NAMES = ('blue_cup', 'beige_cup')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_planning_scene(source, output):
    root = ET.parse(source).getroot()
    world = root.find('worldbody')
    for item in list(world):
        if item.tag == 'body' and item.get('name', '').startswith('robot_'):
            world.remove(item)
    for tag in ('actuator', 'sensor', 'keyframe', 'contact', 'equality'):
        for item in root.findall(tag):
            root.remove(item)
    assets = root.find('asset')
    for item in list(assets):
        if item.get('name', '').startswith('robot_'):
            assets.remove(item)
    size = root.find('size')
    if size is not None:
        size.attrib.pop('nkey', None)
    root.set('model', 'PLANNING_ONLY_episode10_nesting_altered_initial_poses')
    ET.ElementTree(root).write(output, encoding='unicode')
    full = mujoco.MjModel.from_xml_path(str(source))
    plan = mujoco.MjModel.from_xml_path(str(output))
    checks = []
    for name in NAMES:
        a, b = full.body(name).id, plan.body(name).id
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
            assert np.array_equal(getattr(full, field)[a], getattr(plan, field)[b]), (name, field)
        for g in range(plan.ngeom):
            if plan.geom_bodyid[g] != b:
                continue
            old = full.geom(plan.geom(g).name).id
            for field in ('geom_type', 'geom_pos', 'geom_quat', 'geom_size', 'geom_friction',
                          'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity'):
                assert np.array_equal(getattr(full, field)[old], getattr(plan, field)[g]), (name, field)
        checks.append(name)
    assert np.array_equal(full.opt.gravity, plan.opt.gravity)
    assert full.opt.timestep == plan.opt.timestep
    return plan, checks


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def quat(r):
    return Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]


def upright(matrix, axis):
    current = matrix @ axis
    correction, _ = Rotation.align_vectors(np.array([[0., 0., -1.]]), current[None])
    return correction.as_matrix() @ matrix


def contact_depths(model, data):
    maximum = {'object_object': 0., 'object_table': 0.}
    counts = {'object_object': 0, 'object_table': 0}
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = model.body(int(model.geom_bodyid[g1])).name, model.body(int(model.geom_bodyid[g2])).name
        kind = ('object_object' if b1 in NAMES and b2 in NAMES and b1 != b2 else
                'object_table' if (b1 in NAMES and b2 == 'table') or (b2 in NAMES and b1 == 'table') else None)
        if kind:
            maximum[kind] = max(maximum[kind], max(0., -float(c.dist)))
            counts[kind] += 1
    return maximum, counts


def pose_data(model, data, positions, matrices):
    mujoco.mj_resetData(model, data)
    for i, name in enumerate(NAMES):
        j = int(model.body_jntadr[model.body(name).id])
        adr = int(model.jnt_qposadr[j])
        data.qpos[adr:adr+3] = positions[i]
        data.qpos[adr+3:adr+7] = quat(matrices[i])
    mujoco.mj_forward(model, data)


def snapshot(model, data):
    return {name: {'position_setup_m': data.xpos[model.body(name).id].tolist(),
                   'quaternion_wxyz': data.xquat[model.body(name).id].tolist()}
            for name in NAMES}


def errors(poses, reference_positions, reference_matrices, anchors, axes):
    out = {}
    for i, name in enumerate(NAMES):
        p = np.asarray(poses[name]['position_setup_m'])
        r = rot(poses[name]['quaternion_wxyz'])
        center = p + r @ anchors[i]
        target = reference_positions[i] + reference_matrices[i] @ anchors[i]
        angle = np.degrees(np.arccos(np.clip(np.dot(r @ axes[i], reference_matrices[i] @ axes[i]), -1., 1.)))
        out[name] = {'final_frame_center_error_cm': float(np.linalg.norm(center-target)*100),
                     'final_frame_symmetry_orientation_error_deg': float(angle)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=B/'refinement_20261018/nesting_plan/episode_000010')
    ap.add_argument('--settle-count', type=int, default=12)
    ap.add_argument('--settle-seconds', type=float, default=3.)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = B/'refinement_20261017/block_polish/episode_000010/execution/scene_physics.xml'
    xml = args.output/'PLANNING_ONLY_objects_scene.xml'
    model, checks = build_planning_scene(source, xml)
    data = mujoco.MjData(model)
    manifest = json.loads((B/'episodes/episode_000010/input_manifest.json').read_text())
    geom = json.loads((B/'episodes/episode_000010/metric_geometry.json').read_text())['objects']
    axes = np.asarray([geom[n]['axis_local'] for n in NAMES])
    anchors = np.asarray([geom[n]['anchor_local_m'] for n in NAMES])
    with np.load(B/'episodes/episode_000010/reference_trajectory.npz', allow_pickle=False) as z:
        assert tuple(z['object_names'].astype(str)) == NAMES
        target_p = z['object_positions'][-1].copy()
        target_r = np.asarray([rot(q) for q in z['object_quaternions_wxyz'][-1]])
    vertices = []
    for name in NAMES:
        obj = next(x for x in manifest['objects'] if x['id'] == name)
        paths = [x['source'] for x in obj['meshes'] if x['kind'] == 'collision']
        vertices.append(np.concatenate([trimesh.load(p, force='mesh', process=False).vertices for p in paths]))
    pose_data(model, data, target_p, target_r)
    target_collision, _ = contact_depths(model, data)
    base_r = np.asarray([upright(target_r[i], axes[i]) for i in range(2)])
    base_p = target_p.copy()
    table = model.geom('g_table').id
    assert model.geom_type[table] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert np.allclose(data.geom_xmat[table].reshape(3, 3), np.eye(3))
    table_z = float(data.geom_xpos[table, 2] + model.geom_size[table, 1])
    base_p[0, 2] = table_z - float((vertices[0] @ base_r[0].T)[:, 2].min()) + .0001
    # Match estimated central axes, not arbitrary mesh origins.
    center_blue = base_p[0] + base_r[0] @ anchors[0]
    offsets = [(0., 0.), (.003, 0.), (-.003, 0.), (0., .003), (0., -.003),
               (.003, .003), (.003, -.003), (-.003, .003), (-.003, -.003)]
    candidates = []
    for angle in range(0, 360, 30):
        matrices = base_r.copy()
        matrices[1] = Rotation.from_rotvec([0., 0., np.radians(angle)]).as_matrix() @ matrices[1]
        for dx, dy in offsets:
            positions = base_p.copy()
            positions[1, :2] = (center_blue - matrices[1] @ anchors[1])[:2] + [dx, dy]
            # Descend from clear separation, stop at the first collision encountered.
            previous = None
            boundary = None
            for zheight in np.arange(base_p[0, 2]+.23, base_p[0, 2]-.001, -.002):
                positions[1, 2] = zheight
                pose_data(model, data, positions, matrices)
                depth, _ = contact_depths(model, data)
                if depth['object_object'] > 1e-8:
                    boundary = (zheight, previous)
                    break
                previous = zheight
            if boundary is None or boundary[1] is None:
                continue
            lo, hi = boundary
            for _ in range(14):
                positions[1, 2] = (lo+hi)/2
                pose_data(model, data, positions, matrices)
                depth, _ = contact_depths(model, data)
                if depth['object_object'] > 1e-8:
                    lo = positions[1, 2]
                else:
                    hi = positions[1, 2]
            positions[1, 2] = hi+.0005
            pose_data(model, data, positions, matrices)
            poses = snapshot(model, data)
            err = errors(poses, target_p, target_r, anchors, axes)
            value = max(v['final_frame_center_error_cm'] for v in err.values())
            candidates.append({'spin_degrees': angle, 'axis_offset_xy_m': [dx, dy],
                'first_contact_beige_origin_z_m': hi,
                'contact_free_pose': poses, 'errors_to_original_final': err,
                'worst_final_center_error_cm': value})
    candidates.sort(key=lambda row: row['worst_final_center_error_cm'])
    (args.output/'static_candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    print(json.dumps({'static_candidates': len(candidates), 'best': candidates[0] if candidates else None}), flush=True)
    settles = []
    for index, candidate in enumerate(candidates[:args.settle_count]):
        positions = np.asarray([candidate['contact_free_pose'][n]['position_setup_m'] for n in NAMES])
        matrices = np.asarray([rot(candidate['contact_free_pose'][n]['quaternion_wxyz']) for n in NAMES])
        positions[1, 2] += .003  # a genuine short gravity drop above first wall contact
        pose_data(model, data, positions, matrices)
        positions_history, quats_history, velocities, times, penetration = [], [], [], [], []
        for step in range(round(args.settle_seconds/model.opt.timestep)+1):
            if step:
                mujoco.mj_step(model, data)
            if step % 5 == 0:
                mujoco.mj_forward(model, data)
                positions_history.append([data.xpos[model.body(n).id].copy() for n in NAMES])
                quats_history.append([data.xquat[model.body(n).id].copy() for n in NAMES])
                velocities.append(data.qvel.copy()); times.append(data.time)
                penetration.append(contact_depths(model, data)[0])
        final = snapshot(model, data)
        err = errors(final, target_p, target_r, anchors, axes)
        last = np.asarray(times) >= args.settle_seconds-.5
        ph = np.asarray(positions_history)
        drift = np.linalg.norm(ph[last]-ph[last][0], axis=-1).max(axis=0)
        speeds = np.asarray(velocities)[last]
        angles = [float(np.degrees(np.arccos(np.clip(np.dot(rot(final[n]['quaternion_wxyz'])@axes[i], [0.,0.,-1.]),-1,1)))) for i,n in enumerate(NAMES)]
        stable = bool(np.all(drift < .001) and np.max(np.abs(speeds)) < .05)
        filename = f'PLANNING_ONLY_settle_{index:03d}.npz'
        np.savez_compressed(args.output/filename, planning_only=np.array(True),
            altered_initial_poses=np.array(True), object_names=np.asarray(NAMES),
            timestamps_seconds=times, object_positions=ph,
            object_quaternions_wxyz=np.asarray(quats_history), qvel=np.asarray(velocities))
        row = {'candidate_index': index, 'input': candidate,
            'final_pose': final, 'errors_to_original_final': err,
            'last_half_second_position_drift_mm': (drift*1000).tolist(),
            'last_half_second_max_abs_qvel': float(np.max(np.abs(speeds))),
            'final_upright_tilt_degrees': angles, 'settled_by_declared_criterion': stable,
            'criterion': 'last 0.5 s: each body-origin drift <1 mm and every generalized velocity magnitude <0.05 SI units',
            'final_penetration_m': penetration[-1], 'recording': str(args.output/filename)}
        settles.append(row)
        print(json.dumps({'settled_candidate': index, 'stable': stable, 'errors': err, 'tilt': angles}), flush=True)
    best = sorted(settles, key=lambda row:(not row['settled_by_declared_criterion'],
        max(v['final_frame_center_error_cm'] for v in row['errors_to_original_final'].values())))[0] if settles else None
    report = {'purpose': 'PLANNING ONLY: static cup fit and altered-initial-state gravity settling. Not an episode evaluation or task success.',
        'episode': 'episode_000010', 'original_inputs_unchanged': True,
        'official_score': None, 'reference_proxy_mp_sr': None, 'submission_ready': False,
        'source_scene': str(source), 'source_scene_sha256': sha(source),
        'planning_scene': str(xml), 'planning_scene_sha256': sha(xml),
        'object_physics_preserved_checks': checks,
        'source_reference_sha256': sha(B/'episodes/episode_000010/reference_trajectory.npz'),
        'table_top_setup_z_m': table_z, 'original_final_pose_collision_m': target_collision,
        'original_final_positions_setup_m': target_p.tolist(),
        'original_final_quaternions_wxyz': [quat(r).tolist() for r in target_r],
        'static_candidate_count': len(candidates), 'settles': settles, 'best_settled': best,
        'elapsed_seconds': time.monotonic()-started,
        'limitations': ['Finite upright/coaxial grid, not a proof of global geometric impossibility.',
            'Final-frame pose differences are not full-horizon MP-SR.',
            'Object-only gravity settling proves neither robot graspability nor achievable manipulation.',
            'Symmetry axes and centers come from reconstructed assets, not confidential scans.']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
