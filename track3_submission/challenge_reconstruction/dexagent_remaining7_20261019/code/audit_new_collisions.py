#!/usr/bin/env python3
"""Read-only test of supplied convex collisions against reference object poses."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
BASE = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
sys.path.insert(0, str(BASE / 'audit_python'))
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_of(urdf, element):
    mesh = element.find('geometry/mesh')
    path = urdf.parent / mesh.attrib['filename']
    model = trimesh.load(path, force='mesh', process=False)
    model.vertices *= np.fromstring(mesh.attrib.get('scale', '1 1 1'), sep=' ')
    origin = element.find('origin')
    if origin is not None:
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        model.vertices = model.vertices @ rotation.T + np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
    return model, path


def load_object(directory, name):
    urdf = directory / 'assets' / (name + '.urdf')
    root = ET.parse(urdf).getroot()
    visual, visual_path = mesh_of(urdf, root.find('.//visual'))
    parts = []
    for element in root.findall('.//collision'):
        mesh, path = mesh_of(urdf, element)
        hull = ConvexHull(np.asarray(mesh.vertices))
        parts.append(dict(path=str(path), sha256=sha(path), bounds=mesh.bounds,
                          equations=hull.equations, volume=float(hull.volume)))
    return dict(visual=visual, visual_path=str(visual_path), parts=parts,
                visual_sha256=sha(visual_path), urdf_sha256=sha(urdf))


def penetration(points, parts):
    depth = np.zeros(len(points))
    owner = np.full(len(points), -1, dtype=int)
    for index, part in enumerate(parts):
        mask = np.all(points >= part['bounds'][0], axis=1) & np.all(points <= part['bounds'][1], axis=1)
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        eq = part['equations']
        candidate = -np.max(points[indices] @ eq[:, :3].T + eq[:, 3], axis=1)
        better = candidate > depth[indices]
        depth[indices[better]] = candidate[better]
        owner[indices[better]] = index
    return depth, owner


def sample(mesh, count=768):
    rng = np.random.default_rng(20261019)
    vertices = np.asarray(mesh.vertices)
    corners = vertices[rng.choice(len(vertices), min(count, len(vertices)), replace=False)]
    centers = np.asarray(mesh.triangles_center)
    faces = centers[rng.choice(len(centers), min(count, len(centers)), replace=False)]
    return np.vstack([corners, faces])


def detail(points, depth, owner, model):
    inside = np.flatnonzero(depth > .002)
    if not len(inside):
        return dict(points_inside_over_2mm=0)
    selected = inside[np.argsort(depth[inside])[-min(400, len(inside)):]]
    selected_points = points[selected]
    _, distance, _ = trimesh.proximity.closest_point(model['visual'], selected_points)
    # A collision-interior point with a clear upward ray to open space cannot
    # be interpreted as normal surface contact with the visible vessel walls.
    direction = np.broadcast_to(np.array([0., 1., 0.]), selected_points.shape)
    hits = model['visual'].ray.intersects_any(selected_points + direction * 1e-6, direction)
    cavities = (distance > .005) & ~hits
    candidates = np.flatnonzero(cavities)
    examples = []
    for j in candidates[np.argsort(depth[selected[candidates]])[-5:]]:
        i = selected[j]
        examples.append(dict(point_body_m=points[i].tolist(), convex_depth_mm=float(depth[i] * 1000),
            visual_surface_distance_mm=float(distance[j] * 1000), collision_part=int(owner[i]),
            unobstructed_body_positive_y_ray=True))
    result = dict(points_inside_over_2mm=int(len(inside)), tested_deepest_points=len(selected),
        max_convex_depth_mm=float(depth.max() * 1000),
        max_visual_clearance_inside_mm=float(distance.max() * 1000),
        collision_interior_points_clear_to_positive_y_and_visual_clearance_over_5mm=int(cavities.sum()),
        examples=examples)
    if model['visual'].is_watertight:
        outside = ~model['visual'].contains(selected_points)
        result['points_provably_outside_watertight_visual_inside_collision'] = int(outside.sum())
    return result


def scan_pair(models, transforms, first, second, frames, fps):
    probe = sample(models[first]['visual'])
    scans = []
    records = {}
    for frame in frames:
        transform = np.linalg.inv(transforms[frame, second]) @ transforms[frame, first]
        points = probe @ transform[:3, :3].T + transform[:3, 3]
        depth, owner = penetration(points, models[second]['parts'])
        scans.append(dict(frame=int(frame), seconds=float(frame / fps),
            maximum_inside_depth_mm=float(depth.max() * 1000),
            fraction_surface_probes_inside_over_2mm=float(np.mean(depth > .002))))
        records[int(frame)] = (points, depth, owner)
    worst = max(scans, key=lambda row: row['maximum_inside_depth_mm'])
    details = {}
    for frame in sorted({frames[0], frames[-1], worst['frame']}):
        points, depth, owner = records[int(frame)]
        details[str(frame)] = dict(seconds=float(frame / fps), **detail(points, depth, owner, models[second]))
    crossings = [row for row in scans if row['maximum_inside_depth_mm'] > 5.]
    return dict(probe_count=len(probe), worst=worst, first_sampled_intrusion_over_5mm=crossings[0] if crossings else None,
                final=scans[-1], inspected_frames=details, timeline=scans)


def cavity_scan(model):
    bounds = model['visual'].bounds
    # Interior grid, including hollow holder regions. This is a deterministic
    # probe set, not a volume estimate or an automatic replacement collider.
    axes = [np.linspace(bounds[0, i], bounds[1, i], 23)[1:-1] for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    depth, owner = penetration(points, model['parts'])
    return detail(points, depth, owner, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, nargs='+', default=[14, 15, 16, 20, 25, 28, 30])
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    root = BASE / 'source/final_recon_ours'
    report = dict(schema='reference_collision_intrusion_audit_v1', source_root=str(root),
        method='Sample visual vertices and triangle centroids at unchanged source object poses; measure penetration into the supplied union of convex hulls, exact nearest visual-triangle distance, and visibility to positive body Y.',
        units='millimetres for reported distances; original meshes/poses use metres',
        warning='For non-watertight visual meshes, upward-ray evidence identifies open-space intrusion but is not a signed solid-volume proof. Sampled penetration is a lower bound, not an exact pairwise penetration depth.',
        inputs_changed=False, source_visuals_changed=False, source_object_poses_changed=False,
        episodes={})
    for episode in args.episodes:
        directory = root / f'episode_{episode:06d}'
        with np.load(directory / 'object_poses.npz', allow_pickle=False) as data:
            transforms, names = data['world_from_object'], data['names'].tolist()
        models = [load_object(directory, name) for name in names]
        frames = sorted(set(range(0, len(transforms), args.stride)) | {len(transforms) - 1})
        result = dict(frames=len(transforms), fps=20., object_poses_sha256=sha(directory / 'object_poses.npz'),
                      objects={}, pairs={})
        for name, model in zip(names, models):
            result['objects'][name] = dict(visual_path=model['visual_path'], visual_sha256=model['visual_sha256'],
                visual_watertight=bool(model['visual'].is_watertight), urdf_sha256=model['urdf_sha256'],
                collision_parts=len(model['parts']), visual_bounds_m=model['visual'].bounds.tolist(),
                collision_volume_sum_cm3=sum(part['volume'] for part in model['parts']) * 1e6,
                cavity_grid_audit=cavity_scan(model),
                parts=[{k: v for k, v in part.items() if k not in ('equations', 'bounds')}
                       for part in model['parts']])
        for first in range(len(names)):
            for second in range(len(names)):
                if first != second:
                    result['pairs'][names[first] + '_inside_' + names[second]] = scan_pair(models, transforms, first, second, frames, 20.)
        report['episodes'][str(episode)] = result
        (BASE / 'review/collision_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(episode=episode, pairs={key: value['worst'] for key, value in result['pairs'].items()})), flush=True)


if __name__ == '__main__':
    main()
