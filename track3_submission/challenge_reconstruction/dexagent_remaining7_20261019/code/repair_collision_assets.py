#!/usr/bin/env python3
"""Version collision-only vessel/rack repairs with measured cavity validation.

Reconstruction visual vertices, object targets, URDF inertias and masses remain
unchanged. Generated convex meshes are actual surface-derived CoACD parts, not
forces, welds or changes to reference motion. Hand reconstruction is independent.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import audit_new_collisions as audit

BASE = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
REPAIRS = {14: ['plastic_dish_rack', 'blue_cup'], 15: ['plastic_dish_rack'],
           16: ['plastic_dish_rack'], 20: ['blue_cup', 'water_pitcher'],
           25: [], 28: ['plastic_dish_rack', 'white_pot'], 30: ['plastic_dish_rack']}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_parts(paths):
    parts, meshes = [], []
    for path in paths:
        mesh = trimesh.load(path, force='mesh', process=False)
        hull = ConvexHull(np.asarray(mesh.vertices))
        if hull.volume <= 1e-14:
            raise ValueError('A collision part has no physical volume: ' + str(path))
        meshes.append(mesh)
        parts.append(dict(path=str(path), sha256=sha(path), bounds=mesh.bounds,
                          equations=hull.equations, volume=float(hull.volume)))
    return parts, meshes


def free_space_probes(model):
    bounds = model['visual'].bounds
    axes = [np.linspace(bounds[0, i], bounds[1, i], 25)[1:-1] for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    _, distance, _ = trimesh.proximity.closest_point(model['visual'], points)
    points = points[distance > .005]
    visible = np.zeros(len(points), dtype=bool)
    for vector in np.vstack([np.eye(3), -np.eye(3)]):
        indices = np.flatnonzero(~visible)
        if not len(indices):
            break
        directions = np.broadcast_to(vector, (len(indices), 3))
        hits = model['visual'].ray.intersects_any(points[indices] + directions * 1e-6, directions)
        visible[indices[~hits]] = True
    return points[visible]


def target_pair_probes(directory, name, source_model, parts):
    with np.load(directory / 'object_poses.npz', allow_pickle=False) as data:
        transforms, names = data['world_from_object'], data['names'].tolist()
    recipient = names.index(name)
    records = {}
    for giver, other in enumerate(names):
        if giver == recipient:
            continue
        model = audit.load_object(directory, other)
        vertices = audit.sample(model['visual'], 384)
        points = []
        frames = sorted(set(range(0, len(transforms), 5)) | {len(transforms) - 1})
        timeline = []
        for frame in frames:
            relative = np.linalg.inv(transforms[frame, recipient]) @ transforms[frame, giver]
            actual = vertices @ relative[:3, :3].T + relative[:3, 3]
            old_depth, _ = audit.penetration(actual, source_model['parts'])
            new_depth, _ = audit.penetration(actual, parts)
            timeline.append(dict(frame=frame, seconds=frame / 20.,
                old_max_mm=float(old_depth.max() * 1000), new_max_mm=float(new_depth.max() * 1000)))
            points.extend(actual[old_depth > .002])
        if points:
            points = np.unique(np.round(np.asarray(points), 7), axis=0)
            _, distance, _ = trimesh.proximity.closest_point(source_model['visual'], points)
            points = points[distance > .005]
            if len(points):
                directions = np.broadcast_to([0., 1., 0.], points.shape)
                hits = source_model['visual'].ray.intersects_any(points + directions * 1e-6, directions)
                points = points[~hits]
        points = np.asarray(points).reshape(-1, 3)
        new_depth, _ = audit.penetration(points, parts)
        records[other] = dict(clear_reference_surface_probes=len(points),
            max_false_collision_remaining_mm=float(new_depth.max(initial=0) * 1000),
            blocked_over_1_5mm=int(np.sum(new_depth > .0015)), timeline=timeline)
    return records


def validate(directory, name, source_model, parts, meshes):
    free = free_space_probes(source_model)
    new_depth, _ = audit.penetration(free, parts)
    old_depth, _ = audit.penetration(free, source_model['parts'])
    samples = audit.sample(source_model['visual'], 2500)
    inside, _ = audit.penetration(samples, parts)
    joined = trimesh.util.concatenate(meshes)
    _, distance, _ = trimesh.proximity.closest_point(joined, samples)
    distance[inside > 0] = 0.
    bounds_error = np.maximum(joined.bounds[1] - source_model['visual'].bounds[1],
                               source_model['visual'].bounds[0] - joined.bounds[0])
    pairs = target_pair_probes(directory, name, source_model, parts)
    report = dict(collision_part_count=len(parts), volume_sum_cm3=sum(p['volume'] for p in parts) * 1e6,
        minimum_part_volume_m3=min(p['volume'] for p in parts), physical_thickness_nonzero=True,
        free_space_probe_count=len(free), old_max_false_penetration_mm=float(old_depth.max(initial=0) * 1000),
        new_max_false_penetration_mm=float(new_depth.max(initial=0) * 1000),
        new_free_probes_blocked_over_1_5mm=int(np.sum(new_depth > .0015)),
        visual_surface_coverage_probes=len(samples), coverage_max_missing_distance_mm=float(distance.max() * 1000),
        coverage_p99_missing_distance_mm=float(np.percentile(distance, 99) * 1000),
        coverage_fraction_within_1mm=float(np.mean(distance <= .001)),
        bounds_expansion_max_mm=float(np.maximum(bounds_error, 0).max() * 1000),
        target_pair_probes=pairs,
        limitations='Deterministic surface/grid probes do not prove exact global Hausdorff distance. Non-watertight cavity evidence uses exact triangle clearance and unobstructed axis rays; true surface intersections may remain in the supplied reconstruction.')
    report['passed'] = bool(report['new_free_probes_blocked_over_1_5mm'] == 0 and
        report['coverage_max_missing_distance_mm'] <= 1. and report['bounds_expansion_max_mm'] <= 2. and
        all(value['blocked_over_1_5mm'] == 0 for value in pairs.values()))
    return report


def generate(args):
    directory = args.source_root / f'episode_{args.episode:06d}'
    if args.object not in REPAIRS[args.episode]:
        raise ValueError('This object is outside the collision repair scope')
    model = audit.load_object(directory, args.object)
    destination = args.library / f'episode_{args.episode:06d}' / args.object
    if (destination / 'completed.json').exists():
        previous = read(destination / 'completed.json')
        if previous['source_visual_sha256'] != model['visual_sha256']:
            raise ValueError('Existing collision library belongs to another source visual')
        print(json.dumps(previous), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    parts_directory = destination / 'parts'
    if parts_directory.exists():
        raise FileExistsError('Inspect previous partial collision attempt: ' + str(parts_directory))
    parts_directory.mkdir()
    begin = time.monotonic()
    settings = dict(threshold=args.threshold, max_convex_hull=args.max_hulls, preprocess_mode='auto',
        preprocess_resolution=100, resolution=2000, mcts_nodes=10, mcts_iterations=50,
        mcts_max_depth=3, merge=True, decimate=False, seed=44)
    if args.reuse_parts:
        generation = dict(method='validated existing CoACD candidate', candidate_directory=str(args.reuse_parts),
            candidate_report=read(args.reuse_parts / 'report.json') if (args.reuse_parts / 'report.json').exists() else None)
        for index, path in enumerate(sorted(args.reuse_parts.glob('part_*.obj'))):
            shutil.copy2(path, parts_directory / f'part_{index:03d}.obj')
    else:
        import coacd
        generated = coacd.run_coacd(coacd.Mesh(np.asarray(model['visual'].vertices),
                                               np.asarray(model['visual'].faces)), **settings)
        generation = dict(method='CoACD collision-aware convex decomposition',
            package_version=importlib.metadata.version('coacd'), settings=settings,
            documentation='https://github.com/SarahWeiii/CoACD')
        for index, (vertices, faces) in enumerate(generated):
            trimesh.Trimesh(vertices, faces, process=False).export(parts_directory / f'part_{index:03d}.obj')
    paths = sorted(parts_directory.glob('part_*.obj'))
    if not paths:
        raise ValueError('No convex collision meshes generated')
    parts, meshes = load_parts(paths)
    validation = validate(directory, args.object, model, parts, meshes)
    write(destination / 'validation.json', validation)
    if not validation['passed']:
        raise ValueError('Collision candidate fails surface/cavity validation: ' + str(destination / 'validation.json'))
    original_urdf = directory / 'assets' / (args.object + '.urdf')
    tree = ET.parse(original_urdf)
    link = tree.getroot().find('link')
    invariant = {key: ET.tostring(link.find(key)) for key in ['inertial', 'visual']}
    visual = link.find('visual')
    origin = visual.find('origin')
    if origin is not None and (origin.get('xyz', '0 0 0') != '0 0 0' or origin.get('rpy', '0 0 0') != '0 0 0'):
        raise ValueError('Non-identity visual body origin requires an explicit frame conversion')
    if visual.find('geometry/mesh').get('scale', '1 1 1') != '1 1 1':
        raise ValueError('Non-unit visual scale requires an explicit frame conversion')
    for old in list(link.findall('collision')):
        link.remove(old)
    for path in paths:
        collision = ET.SubElement(link, 'collision')
        geometry = ET.SubElement(collision, 'geometry')
        ET.SubElement(geometry, 'mesh', filename=f'{args.object}_convex_repaired/{path.name}')
    if any(ET.tostring(link.find(key)) != value for key, value in invariant.items()):
        raise ValueError('Repair changed visual or inertial URDF fields')
    output_urdf = destination / (args.object + '.urdf')
    ET.indent(tree)
    tree.write(output_urdf, encoding='utf-8', xml_declaration=True)
    report = dict(schema='collision_only_coacd_repair_v1', episode=args.episode, object=args.object,
        source_directory=str(directory), source_visual_path=model['visual_path'],
        source_visual_sha256=model['visual_sha256'], source_urdf_sha256=sha(original_urdf),
        source_object_poses_sha256=sha(directory / 'object_poses.npz'),
        repaired_urdf=dict(path=str(output_urdf), sha256=sha(output_urdf)),
        convex_parts=[dict(path=str(path), sha256=sha(path)) for path in paths], generation=generation,
        validation=validation, visual_mesh_unchanged=True, body_frame_unchanged=True,
        mass_inertia_unchanged=True, reference_object_poses_unchanged=True,
        hand_reconstruction_independent=True, script_sha256=sha(__file__),
        elapsed_seconds=time.monotonic() - begin, physical_training_not_yet_run=True)
    write(destination / 'completed.json', report)
    print(json.dumps(dict(episode=args.episode, object=args.object, output=str(destination),
        parts=len(parts), passed=True, elapsed_seconds=report['elapsed_seconds'])), flush=True)


def merge(args):
    name = f'episode_{args.episode:06d}'
    source = args.source_root / name
    output = args.output_root / name
    if output.exists():
        raise FileExistsError('Preserve an already copied hand/collision version: ' + str(output))
    reports = {name: read(args.library / f'episode_{args.episode:06d}' / name / 'completed.json')
               for name in REPAIRS[args.episode]}
    for object_name, report in reports.items():
        visual = Path(report['source_visual_path']).name
        if sha(source / 'assets' / visual) != report['source_visual_sha256']:
            raise ValueError('Hand-rebased input changed the visual body geometry')
        if sha(source / 'object_poses.npz') != report['source_object_poses_sha256']:
            raise ValueError('Hand-rebased input changed the reference object targets')
        if not report['validation']['passed']:
            raise ValueError('A collision library component has not passed validation')
    shutil.copytree(source, output,
        ignore=shutil.ignore_patterns('archive_wilor', '*.mp4', '__pycache__', '*.pyc'))
    for object_name, report in reports.items():
        original = output / 'assets' / (object_name + '.urdf')
        old = ET.parse(original).getroot().find('link')
        replacement = Path(report['repaired_urdf']['path'])
        new = ET.parse(replacement).getroot().find('link')
        for tag in ['visual', 'inertial']:
            # Serialization whitespace may differ; compare all attributes and
            # child structure, which are the complete physical/visual values.
            signature = lambda node: (node.tag, dict(node.attrib), [signature(child) for child in node])
            if signature(old.find(tag)) != signature(new.find(tag)):
                raise ValueError('Repair would alter preserved ' + tag + ': ' + object_name)
        shutil.copy2(replacement, original)
        destination = output / 'assets' / (object_name + '_convex_repaired')
        destination.mkdir()
        for part in report['convex_parts']:
            if sha(part['path']) != part['sha256']:
                raise ValueError('Validated collision library changed')
            shutil.copy2(part['path'], destination / Path(part['path']).name)
    immutable = ['reference.parquet', 'object_poses.npz', 'camera_motion.npz', 'hand_report.json',
                 'hand_targets.npz', 'hand_validity.npz', 'sharpa_ik.npz']
    preserved = {key: sha(source / key) for key in immutable if (source / key).is_file()}
    if any(sha(output / key) != value for key, value in preserved.items()):
        raise ValueError('Hand/reference input bytes changed while merging collision assets')
    write(output / 'collision_repair_provenance.json', dict(schema='hand_and_collision_version_v1',
        episode=args.episode, source_hand_reconstruction=str(source),
        preserved_files=preserved, repairs=reports, visual_meshes_unchanged=True,
        masses_inertias_unchanged=True, body_frames_unchanged=True, object_targets_unchanged=True,
        collision_files_only_changed=True, physical_training_not_yet_run=True))
    print(json.dumps(dict(output=str(output), repaired_objects=list(reports),
                         source_hand_reconstruction=str(source))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'merge'])
    parser.add_argument('--episode', type=int, choices=sorted(REPAIRS), required=True)
    parser.add_argument('--object')
    parser.add_argument('--source-root', type=Path, default=BASE / 'hawor_recon')
    parser.add_argument('--library', type=Path, default=BASE / 'collision_library')
    parser.add_argument('--output-root', type=Path, default=BASE / 'collision_recon')
    parser.add_argument('--threshold', type=float, default=.03)
    parser.add_argument('--max-hulls', type=int, default=64)
    parser.add_argument('--reuse-parts', type=Path)
    arguments = parser.parse_args()
    generate(arguments) if arguments.mode == 'generate' else merge(arguments)
