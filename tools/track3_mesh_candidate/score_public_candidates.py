"""Score video-derived public candidates against released reference meshes.

Selecting the minimum on this same public set is an in-sample upper bound,
not an unbiased estimate of hidden leaderboard performance.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('public_mesh_score', HERE / 'score.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
PUBLIC = Path('/data1/video_to_data_challenge/track_3/public')


def pairs():
    result = {}
    for parquet in sorted((PUBLIC / 'data/chunk-000').glob('episode_*.parquet')):
        episode = int(parquet.stem.split('_')[-1])
        objects = pq.read_table(parquet, columns=['observation.objects']).column(0)[0].as_py()
        for slot, obj in enumerate(objects):
            result.setdefault(obj['name'], []).append((episode, slot))
    return result


def main():
    pair_map = pairs()
    all_rows = []
    best = {}
    for item in json.loads((ROOT / 'objects.json').read_text()):
        name = item['name']
        reference = scorer.mesh_arrays(scorer.reference_path(name))
        paths = [ROOT / 'meshes' / name / 'generated.glb']
        paths.extend(sorted((ROOT / 'candidates' / name).glob('*/generated.glb')))
        paths.extend(sorted((ROOT / 'mvsam3d' / 'visualization' / 'input' / name)
                            .glob('*/result.glb')))
        if not paths:
            raise FileNotFoundError(f'no meshes for {name}')
        scores = []
        for path in paths:
            candidate = scorer.mesh_arrays(path)
            values = [scorer.object_chamfer_cm(reference, candidate,
                      scorer.object_seed(episode, slot))
                      for episode, slot in pair_map[name]]
            mean = float(np.mean(values))
            all_rows.append({'object_name': name, 'mesh': str(path),
                             'pairs': len(values), 'mean_cd_o_cm': mean})
            scores.append((mean, path))
            print(name, path.parent.name, f'{mean:.6f} cm', flush=True)
        mean, path = min(scores)
        best[name] = {'mesh': str(path), 'pairs': len(pair_map[name]),
                      'mean_cd_o_cm': mean}
    with (ROOT / 'candidate_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['object_name', 'mesh',
                                                     'pairs', 'mean_cd_o_cm'])
        writer.writeheader()
        writer.writerows(all_rows)
    n = sum(x['pairs'] for x in best.values())
    weighted = sum(x['pairs'] * x['mean_cd_o_cm'] for x in best.values()) / n
    report = {'metric': 'official public CD-O, cm, Sim(3)',
              'selection': 'in-sample minimum over video-derived SAM3D and MV-SAM3D candidates',
              'pairs': n, 'mean_cd_o_cm': weighted, 'objects': best}
    (ROOT / 'candidate_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
