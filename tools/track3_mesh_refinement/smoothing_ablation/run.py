"""Fixed, GT-independent surface smoothing; public GT is opened only for scoring."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from score import mesh_arrays, reference_path, object_chamfer_cm, object_seed
from v2dlb.mesh_budget import budget_mesh


def main():
    selection = json.loads((ROOT.parent / 'selected_video_only/selection_manifest.json').read_text())
    names = sorted(selection['objects'])
    name = names[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    source = Path(selection['objects'][name]['mesh'])
    packed = '--packed' in sys.argv
    out = ROOT / 'packed' / name if packed else ROOT / name
    out.mkdir(exist_ok=True, parents=True)
    mesh = trimesh.load(source, force='mesh', process=False)
    variants = {'original': source}
    # These settings are fixed for every object; no reference geometry enters generation.
    for iterations in (5, 15, 30):
        candidate = mesh.copy()
        trimesh.smoothing.filter_taubin(candidate, lamb=0.5, nu=0.53, iterations=iterations)
        if not np.isfinite(candidate.vertices).all():
            raise ValueError('Non-finite smoothed vertices')
        dest = out / f'taubin_{iterations}.glb'
        candidate.export(dest)
        variants[f'taubin_{iterations}'] = dest
    (out / 'provenance.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'generation_uses_gt': False, 'variants': {k: str(v) for k, v in variants.items()},
    }, indent=2))
    if packed:
        sample = pq.read_table('/data1/video_to_data_challenge/v2d_submission_kit/data/track_3_mesh_sample_submission.parquet', columns=['row_id']).column(0).to_pylist()
        prefix = sample[0].rsplit('_', 2)[0] + '_'
        vertices_budget = sum(r.startswith(prefix + '0_') for r in sample)
        faces_budget = sum(r.startswith(prefix + '1_') for r in sample)
        if not vertices_budget or not faces_budget:
            raise ValueError('Invalid sample budget')
        load = lambda p: budget_mesh(p, faces_budget, vertices_budget)
    else:
        load = mesh_arrays
    reference = load(reference_path(name))
    with (ROOT.parent / 'cd_o_per_episode_object.csv').open() as handle:
        roster = [r for r in csv.DictReader(handle) if r['object_name'] == name]
    rows = []
    for variant, path in variants.items():
        candidate = load(path)
        for entry in roster:
            episode, slot = int(entry['episode']), int(entry['object_slot'])
            value = object_chamfer_cm(reference, candidate, object_seed(episode, slot))
            row = dict(object_name=name, variant=variant, episode=episode, slot=slot, cd_o_cm=value)
            rows.append(row)
            print(json.dumps(row), flush=True)
            temporary = out / 'scores.tmp.json'
            temporary.write_text(json.dumps(rows, indent=2))
            temporary.replace(out / 'scores.json')
    means = {v: float(np.mean([r['cd_o_cm'] for r in rows if r['variant'] == v])) for v in variants}
    (out / 'completed.json').write_text(json.dumps({'means_cm': means, 'pairs': len(roster), 'packed': packed}, indent=2))


if __name__ == '__main__':
    main()
