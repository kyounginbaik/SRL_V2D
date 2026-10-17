#!/usr/bin/env python3
"""Read-only collision audit of unchanged source object poses at original times.

Uses the exact experiment colliders and support geometry. No stepping, object
target correction, pose fitting, or manipulation-success score is performed.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from grasp_stability import _body_lookup, mask_summary, sha256

B = Path(__file__).resolve().parents[1]


def depth_summary(depth, times):
    depth = np.asarray(depth, float)
    positive = depth > 0
    weights = np.r_[np.diff(times), 0.]
    i = int(np.argmax(depth))
    return dict(max_penetration_mm=float(depth[i]*1000),
                time_of_max_s=float(times[i]),
                mean_frame_max_penetration_mm=float(depth.mean()*1000),
                time_weighted_mean_frame_max_penetration_mm=float(np.dot(weights, depth)/weights.sum()*1000),
                mean_over_penetrating_frames_mm=float(depth[positive].mean()*1000) if positive.any() else 0.,
                thresholds={str(mm): mask_summary(depth > mm/1000, times, True)
                            for mm in (0, 1, 5, 10, 20, 30)},
                frame_max_penetration_mm=(depth*1000).tolist())


def audit(scene, manifest_path):
    scene, manifest_path = Path(scene).resolve(), Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['source'])/'object_poses.npz'
    if sha256(source) != manifest['source_object_poses_sha256']:
        raise ValueError('Source poses changed after scene preparation')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0
    with np.load(source, allow_pickle=False) as z:
        names = z['names'].astype(str).tolist()
        poses = np.asarray(manifest['setup_from_source']) @ z['world_from_object']
    if poses.shape != (manifest['frames'], len(names), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Source pose dimensions/values do not match the full episode')
    times = np.arange(len(poses))/float(manifest['fps'])
    roots, by_body, _ = _body_lookup(model, names)
    addresses = {}
    for name, body in roots.items():
        joint = int(model.body_jntadr[body])
        if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f'{name} must be a root free-joint body')
        if model.body_parentid[body] != 0:
            raise ValueError(f'{name} free-joint pose is not expressed in the setup frame')
        addresses[name] = int(model.jnt_qposadr[joint])
    static = {g for g in range(model.ngeom) if model.body_weldid[model.geom_bodyid[g]] == 0
              and (model.geom_contype[g] or model.geom_conaffinity[g])}
    table = {g for g in static if model.geom(g).name == 'g_table'}
    if not table:
        raise ValueError('Expected explicit finite support table geom g_table')
    pairs = {f'{a}__{b}': np.zeros(len(times)) for a,b in itertools.combinations(names, 2)}
    table_depth = {name: np.zeros(len(times)) for name in names}
    records = []
    max_pose_difference = 0.
    for i, frame in enumerate(poses):
        for name, pose in zip(names, frame):
            address = addresses[name]
            data.qpos[address:address+3] = pose[:3, 3]
            data.qpos[address+3:address+7] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        for name, pose in zip(names, frame):
            max_pose_difference = max(max_pose_difference,
                float(np.max(np.abs(data.xpos[roots[name]]-pose[:3, 3]))),
                float(np.max(np.abs(data.xmat[roots[name]].reshape(3, 3)-pose[:3, :3]))))
        deepest = None
        for k in range(data.ncon):
            c = data.contact[k]
            depth = max(0., -float(c.dist))
            if depth <= 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            o1, o2 = by_body.get(int(model.geom_bodyid[g1])), by_body.get(int(model.geom_bodyid[g2]))
            category = None
            if o1 and o2 and o1 != o2:
                a,b = sorted((o1,o2), key=names.index)
                key = f'{a}__{b}'
                pairs[key][i] = max(pairs[key][i], depth)
                category = key
            elif o1 and g2 in table:
                table_depth[o1][i] = max(table_depth[o1][i], depth)
                category = f'{o1}__table'
            elif o2 and g1 in table:
                table_depth[o2][i] = max(table_depth[o2][i], depth)
                category = f'{o2}__table'
            if category and (deepest is None or depth*1000 > deepest['penetration_mm']):
                deepest = dict(time_s=float(times[i]), category=category, penetration_mm=depth*1000,
                               geom1=model.geom(g1).name, geom2=model.geom(g2).name,
                               contact_position_setup_m=c.pos.tolist())
        records.append(deepest)
    if max_pose_difference > 1e-5:
        raise ValueError(f'Model object poses differ from immutable reference: {max_pose_difference}')
    objects = {}
    for name in names:
        geoms = [g for g in range(model.ngeom) if by_body.get(int(model.geom_bodyid[g])) == name
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        objects[name] = dict(collision_geom_count=len(geoms),
            collision_geom_names=[model.geom(g).name for g in geoms],
            mesh_geom_count=int(sum(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in geoms)),
            mass_kg=float(model.body_mass[roots[name]]))
    both = np.maximum.reduce(list(pairs.values())) if pairs else np.zeros(len(times))
    support = np.maximum.reduce(list(table_depth.values()))
    return dict(schema='dexagent.reference_collisions.v1', episode=manifest['episode'],
        scene=str(scene), manifest=str(manifest_path), source_object_poses=str(source),
        hashes=dict(scene=sha256(scene), manifest=sha256(manifest_path), source_poses=sha256(source), audit_code=sha256(__file__)),
        object_names=names, objects=objects, sample_count=len(times), fps=float(manifest['fps']),
        timestamps_seconds=times.tolist(), immutable_pose_max_numeric_difference=max_pose_difference,
        object_object=depth_summary(both, times), object_table=depth_summary(support, times),
        pairs={k: depth_summary(v, times) for k,v in pairs.items()},
        objects_against_table={k: depth_summary(v, times) for k,v in table_depth.items()},
        deepest_contact_per_frame=records,
        static_geometry=[dict(name=model.geom(g).name, type=int(model.geom_type[g]),
                              position=model.geom_pos[g].tolist(), size=model.geom_size[g].tolist()) for g in sorted(static)],
        interpretation=[
          'Penetration is max(0, -MuJoCo contact distance) using unchanged experiment collision filters and convex pieces.',
          'Each frame contributes its maximum pair penetration; frame means include zero-penetration frames.',
          'Hands are ignored in reported contacts. No dynamics integration, force estimate, or object/reference fitting.',
          'This diagnoses inconsistency with these reconstructed colliders and the inferred finite table, not official ground truth.',
          'Reference penetration alone does not prove MP-SR infeasible: mean-error tolerances permit departures from exact poses.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, nargs='+', default=[1,3,5,8,10])
    p.add_argument('--tag', default='cem_v2')
    p.add_argument('--output', type=Path, default=B/'refinement_20261017/reference_collision_audit')
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    for n in a.episodes:
        ep=f'episode_{n:06d}'
        result=json.loads((B/'refinement_20261017'/a.tag/ep/'result.json').read_text())
        report=audit(result['execution']['scene'], B/'episodes'/ep/'input_manifest.json')
        (a.output/f'{ep}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(episode=ep, **{k:{j:report[k][j] for j in ('max_penetration_mm','mean_frame_max_penetration_mm','time_of_max_s')} for k in ('object_object','object_table')})), flush=True)


if __name__=='__main__':
    main()
