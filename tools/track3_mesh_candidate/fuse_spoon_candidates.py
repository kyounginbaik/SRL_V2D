"""Fuse independent video-derived SAM3D spoon surfaces without reference scans."""
import sys
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

KIT = Path('/data1/video_to_data_challenge/v2d_submission_kit')
sys.path.insert(0, str(KIT))
from v2dlb.mesh_common import _sample_mesh_surface  # noqa: E402
from v2dlb.object_mesh import align_rigid, surface_moments  # noqa: E402

ROOT = Path(os.environ.get('V2D_SPOON_CANDIDATE_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/spoon_mesh_candidates'))
NAMES = ['ep000036_f000000_s42', 'ep000035_f000000_s42',
         'ep000037_f000000_s42', 'ep000036_f000260_s42']


def load(name):
    path = ROOT / name / 'generated.glb'
    mesh = trimesh.load(path, force='mesh', process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def main():
    ref_v, ref_f = load(NAMES[0])
    ref_center, ref_radius = surface_moments(ref_v, ref_f)
    ref_align = _sample_mesh_surface(ref_v, ref_f, 3000, 101)
    nearest_surfaces = []
    for name in NAMES[1:]:
        cand_v, cand_f = load(name)
        center, radius = surface_moments(cand_v, cand_f)
        cand_v = (cand_v - center) * (ref_radius / radius) + ref_center
        cand_align = _sample_mesh_surface(cand_v, cand_f, 3000, 101)
        rotation, translation = align_rigid(ref_align, cand_align)
        surface = _sample_mesh_surface(cand_v, cand_f, 120000, 202)
        surface = surface @ rotation.T + translation
        distance, index = cKDTree(surface).query(ref_v, k=1)
        nearest_surfaces.append(surface[index])
        print(name, 'median nearest distance', np.median(distance), flush=True)

    average = np.mean(nearest_surfaces, axis=0)
    for alpha in (0.25, 0.5, 0.75):
        vertices = (1.0 - alpha) * ref_v + alpha * average
        out = ROOT / f'consensus4_alpha{int(alpha * 100):02d}.glb'
        trimesh.Trimesh(vertices=vertices, faces=ref_f, process=False).export(out)
        print(out, flush=True)


if __name__ == '__main__':
    main()
