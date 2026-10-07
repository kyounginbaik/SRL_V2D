"""Compare video-derived spoon meshes against tracked episode 37 masks.

These scores are a shape-selection proxy, not the hidden CD-O metric. The
object poses were fitted with the existing episode mesh, so weakly aligned
frames are excluded from the aggregate.
"""
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, '/data1/video_to_data_challenge/v2d_submission_kit')
from v2dlb.mesh_common import _sample_mesh_surface  # noqa: E402
from v2dlb.object_mesh import align_rigid, surface_moments  # noqa: E402

ROOT = Path(os.environ.get('V2D_CHALLENGE_RECON_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction'))
EPISODE = ROOT / 'evaluation/episode_000037'
SPOONS = ROOT / 'spoon_mesh_candidates'
PATHS = {
    'episode_mesh': EPISODE / 'result/mesh.obj',
    'medoid': SPOONS / 'ep000036_f000000_s42/generated.glb',
    'consensus3_alpha25': SPOONS / 'consensus_alpha25.glb',
    'consensus3_alpha50': SPOONS / 'consensus_alpha50.glb',
    'consensus3_alpha75': SPOONS / 'consensus_alpha75.glb',
    'consensus4': SPOONS / 'consensus4_alpha25.glb',
    'episode37_frame60': SPOONS / 'ep000037_f000060_s42/generated.glb',
}


def load(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def aligned_meshes():
    ref_v, ref_f = load(PATHS['episode_mesh'])
    ref_center, ref_radius = surface_moments(ref_v, ref_f)
    ref_sample = _sample_mesh_surface(ref_v, ref_f, 3000, 101)
    result = {'episode_mesh': (ref_v, ref_f)}
    for name, path in PATHS.items():
        if name == 'episode_mesh':
            continue
        vertices, faces = load(path)
        center, radius = surface_moments(vertices, faces)
        vertices = (vertices - center) * (ref_radius / radius) + ref_center
        sample = _sample_mesh_surface(vertices, faces, 3000, 101)
        rotation, translation = align_rigid(ref_sample, sample)
        result[name] = (vertices @ rotation.T + translation, faces)
    return result


def render(vertices, faces, rotation, translation, camera, shape, scale):
    xyz = vertices @ rotation.T + translation
    uv = np.column_stack((
        xyz[:, 0] / xyz[:, 2] * camera['fx'] + camera['cx'],
        xyz[:, 1] / xyz[:, 2] * camera['fy'] + camera['cy'],
    )) * scale
    uv = np.rint(uv).astype(np.int32)
    valid_faces = faces[np.all(xyz[faces, 2] > 1e-4, axis=1)]
    result = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(result, uv[valid_faces], 1)
    return result > 0


def main():
    camera = json.loads((EPISODE / 'intrinsics/000000.json').read_text())
    scale = 0.25
    shape = (round(camera['height'] * scale), round(camera['width'] * scale))
    meshes = aligned_meshes()
    rows = []
    for frame in range(0, 150, 5):
        pose_file = EPISODE / f'poses_refined_simple/{frame:06d}.json'
        mask_file = EPISODE / f'masks/1/{frame:06d}.png'
        if not pose_file.is_file() or not mask_file.is_file():
            continue
        pose = json.loads(pose_file.read_text())
        # Pipeline JSON quaternions are scalar-first (w, x, y, z).
        wxyz = pose['rotation']
        rotation = Rotation.from_quat(wxyz[1:] + wxyz[:1]).as_matrix()
        translation = np.asarray(pose['translation'])
        observed = cv2.resize(cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE),
                              shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        row = {'frame': frame, 'mask_pixels': int(observed.sum()), 'iou': {}}
        for name, (vertices, faces) in meshes.items():
            predicted = render(vertices, faces, rotation, translation, camera, shape, scale)
            intersection = np.logical_and(predicted, observed).sum()
            union = np.logical_or(predicted, observed).sum()
            row['iou'][name] = float(intersection / union) if union else 0.0
        rows.append(row)
    reliable = [row for row in rows if row['iou']['episode_mesh'] >= 0.4]
    report = {
        'method': 'quarter-resolution silhouette IoU; frames every 5; episode-mesh IoU >= 0.4',
        'total_frames': len(rows),
        'reliable_frames': len(reliable),
        'mean_iou': {name: float(np.mean([row['iou'][name] for row in reliable]))
                     for name in meshes},
        'frames': rows,
    }
    out = SPOONS / 'silhouette_comparison.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
