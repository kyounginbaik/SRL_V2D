"""Compare video-derived spoon candidates by the official shape metric."""
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

SCORE = Path(__file__).resolve().parent / 'score.py'
spec = importlib.util.spec_from_file_location('public_mesh_score', SCORE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ROOT = Path(os.environ.get('V2D_SPOON_CANDIDATE_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/spoon_mesh_candidates'))
PATHS = sorted(ROOT.glob('*/generated.glb'))
PATHS.extend([
    Path('/data1/SRL_V2D/challenge_reconstruction/evaluation/episode_000035/mesh/textured_mesh.obj'),
    Path('/data1/SRL_V2D/challenge_reconstruction/evaluation/episode_000037/mesh/textured_mesh.obj'),
])


def main():
    names = [p.parent.name if p.name == 'generated.glb' else f'{p.parent.parent.name}_{p.parent.name}'
             for p in PATHS]
    meshes = [module.mesh_arrays(path) for path in PATHS]
    # Scale all surfaces to the same 0.15 m RMS radius, so directed score
    # magnitudes can be compared across SAM3D's arbitrary canonical scales.
    normalized = []
    for vertices, faces in meshes:
        center, radius = module.object_chamfer_cm.__globals__['_surface_moments'](vertices, faces)
        normalized.append(((vertices - center) * (0.15 / radius), faces))
    matrix = np.zeros((len(names), len(names)))
    for i, j in itertools.combinations(range(len(names)), 2):
        value = module.object_chamfer_cm(normalized[i], normalized[j], 42)
        matrix[i, j] = matrix[j, i] = value
        print(names[i], names[j], f'{value:.4f} cm', flush=True)
    mean = matrix.sum(axis=1) / (len(names) - 1)
    result = {'names': names, 'matrix_cm': matrix.tolist(),
              'mean_to_others_cm': dict(zip(names, mean.tolist())),
              'medoid': names[int(np.argmin(mean))]}
    (ROOT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('MEDOID', result['medoid'], flush=True)


if __name__ == '__main__':
    main()
