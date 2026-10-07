"""Score generated video-only meshes using the official Track 3 CD-O function."""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh

KIT = Path('/data1/video_to_data_challenge/v2d_submission_kit')
sys.path.insert(0, str(KIT))
from v2dlb.object_mesh import object_chamfer_cm, object_seed  # noqa: E402

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))
PUBLIC = Path('/data1/video_to_data_challenge/track_3/public')


def mesh_arrays(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'{path}: {type(mesh)}')
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError(f'{path}: empty mesh')
    return vertices, faces


def reference_path(name):
    files = list((PUBLIC / 'mesh' / name).glob('*.glb'))
    visual = [p for p in files if not p.name.endswith('_collision.glb')]
    if len(visual) != 1:
        raise ValueError(f'{name}: expected one visual reference, found {visual}')
    return visual[0]


def main():
    entries = json.loads((ROOT / 'objects.json').read_text())
    candidates = {}
    references = {}
    for item in entries:
        name = item['name']
        path = ROOT / 'meshes' / name / 'generated.glb'
        if path.is_file() and path.stat().st_size:
            candidates[name] = mesh_arrays(path)
            references[name] = mesh_arrays(reference_path(name))
        else:
            print(f'MISSING {name}: {path}', flush=True)

    rows = []
    for parquet in sorted((PUBLIC / 'data/chunk-000').glob('episode_*.parquet')):
        episode = int(parquet.stem.split('_')[-1])
        objects = pq.read_table(parquet, columns=['observation.objects']).column(0)[0].as_py()
        for slot, obj in enumerate(objects):
            name = obj['name']
            if name not in candidates:
                continue
            score = object_chamfer_cm(references[name], candidates[name], object_seed(episode, slot))
            row = {'episode': episode, 'object_slot': slot, 'object_name': name,
                   'cd_o_cm': score, 'mesh': str(ROOT / 'meshes' / name / 'generated.glb')}
            rows.append(row)
            print(f'{episode:06d} {slot} {name} {score:.6f} cm', flush=True)

    with (ROOT / 'cd_o_per_episode_object.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'object_slot', 'object_name', 'cd_o_cm', 'mesh'])
        writer.writeheader()
        writer.writerows(rows)
    by_name = {}
    for row in rows:
        by_name.setdefault(row['object_name'], []).append(row['cd_o_cm'])
    summary = {
        'metric': 'official v2dlb.object_mesh.object_chamfer_cm, Sim(3), cm',
        'coverage': {'objects': len(by_name), 'episode_object_pairs': len(rows)},
        'mean_cd_o_cm': float(np.mean([r['cd_o_cm'] for r in rows])) if rows else None,
        'objects': {name: {'pairs': len(values), 'mean_cd_o_cm': float(np.mean(values))}
                    for name, values in sorted(by_name.items())},
    }
    (ROOT / 'cd_o_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
