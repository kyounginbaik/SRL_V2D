"""Plan, execute, and register isolated one-factor reconstruction experiments.

Planning is CPU-only. Execution needs the same Docker images/weights as the
normal ego pipeline and an allocated GPU. No RL or retargeting is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / '.ablation_lab/experiments.json'
BASE = dict(lens='anycalib', depth='moge', camera='slam', mesh='provided',
            pose='foundationpose', hand='hamer', crop='mask', stage='filled', gravity='on', refine='off')
FACTORS = {'hand': ['hamer', 'hawor'], 'refine': ['off', 'on'], 'lens': ['anycalib', 'off'],
           'mesh': ['provided', 'sam3d'], 'gravity': ['on', 'off'], 'camera': ['slam', 'static'],
           'crop': ['mask', 'detector']}


def read_registry(path=REGISTRY):
    return json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'runs': []}


def save(registry, path=REGISTRY):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(registry, indent=2)); tmp.replace(path)


def make_command(episode, config, output, dataset, python):
    rows = [line.split('\t') for line in (dataset / 'runs/prep/episodes.tsv').read_text().splitlines() if not line.startswith('#')]
    row = next(row for row in rows if int(row[0]) == episode)
    obj, prompt = row[2:4]
    command = [python, str(REPO / 'reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py'),
        '--video', str(dataset / f'track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_{episode:06d}.mp4'),
        '--output_dir', str(output), '--object_prompt', prompt, '--hand_tracking', config['hand'], '--reference_frame', '0',
        '--export_threejs_result', '--dev']
    if config['hand'] == 'hamer':
        command += ['--hamer_crop_source', config['crop']]
    if config['mesh'] == 'provided':
        command += ['--object_mesh', str(dataset / f'runs/prep/meshes/{obj}/{obj}.obj'), '--skip_object_scale_estimation']
    for key, val, flag in [('lens','anycalib','--undistort'),('camera','slam','--run_droid_slam'),('gravity','on','--run_gravity_alignment'),('refine','on','--run_gsplat_refinement')]:
        if config[key] == val:
            command.append(flag)
    return command


def validate_result(record):
    root = Path(record['output_dir'])
    config = record['config']
    folder = 'result_slam' if config['camera'] == 'slam' else 'result'
    if config['gravity'] == 'on':
        folder += '_gravity_aligned'
    result = root / folder
    for name in ['result.npz', 'mesh.obj', 'manifest.json']:
        if not (result / name).is_file() or (result / name).stat().st_size == 0:
            raise ValueError(f'Missing output: {result / name}')
    manifest = json.loads((result / 'manifest.json').read_text())
    sources = manifest.get('sources', {})
    if sources.get('hand_pose_source') != config['hand']:
        raise ValueError('Recorded hand model differs from the requested experiment')
    if config['hand'] == 'hamer' and sources.get('hamer_crop_source') != config['crop']:
        raise ValueError('Recorded HaMeR crop source differs from the requested experiment')
    refined = 'refined' in sources.get('final_hand_root', '')
    if refined != (config['refine'] == 'on'):
        raise ValueError('Refinement provenance differs from the requested experiment')
    return result


def register_scene(record, site, dataset):
    # Import only after the experiment is validated, so failed runs never
    # acquire selectable geometry or borrow another variant's numbers.
    from . import build as b
    from .metrics import bundle_checks, trajectory_metrics
    import numpy as np
    import trimesh
    result = validate_result(record)
    epath = site / f'episode_{record["episode"]:06d}/episode.json'
    episode = json.loads(epath.read_text())
    arrays = dict(np.load(result / 'result.npz', allow_pickle=False))
    problems = bundle_checks(arrays, episode['frames'])
    if problems:
        raise ValueError('; '.join(problems))
    b.OUTPUT = site
    exp = b.exporter(); model = exp._load_mano_right_model(dataset / 'runs/mano/mano_v1_2')
    key = 'experiment_' + record['id']
    out = epath.parent / key
    hand = b.export_hands(out / 'hands.bin', b.bundle_hands(arrays), model, exp, episode['fps'])
    mesh = trimesh.load(result / 'mesh.obj', force='mesh', process=False)
    mesh_spec = {'vertices': b.binary(out / 'vertices.bin', mesh.vertices), 'faces': b.binary(out / 'faces.bin', mesh.faces, '<u4')}
    metric, alignment = ({}, None)
    # Different generated mesh frames cannot use provided scan-pose anchors.
    # Centre trajectories are also incomparable without an origin mapping.
    if record['config']['mesh'] == 'provided':
        truth, valid, _ = b.truth_poses(dataset, episode['id'], episode['object'])
        metric, alignment = trajectory_metrics(arrays['object_to_world_transform'], truth, arrays['object_is_valid'] & valid)
    episode['hands'][key] = hand
    episode['variants'][key] = {'camera': arrays['camera_to_world_transform'].round(7).tolist(),
        'object_camera': arrays['object_to_camera_transform'].round(7).tolist(), 'object_valid': arrays['object_is_valid'].tolist(),
        'metrics': metric, 'alignment': alignment, 'mesh': mesh_spec, 'checks': problems,
        'bundle': str(result / 'result.npz'), 'refined': record['config']['refine'] == 'on',
        'mode': 'gravity' if record['config']['gravity'] == 'on' else record['config']['camera']}
    episode.setdefault('experiments', {})[key] = {'config': record['config'], 'hand_key': key, 'variant_key': key,
        'command': record['command'], 'status': 'completed', 'provenance': record.get('provenance', {})}
    b.write_json(epath, episode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--dataset', type=Path, default=Path('/data1/video_to_data_challenge'))
    parser.add_argument('--site', type=Path, default=REPO / '.ablation_lab/site')
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan');plan.add_argument('--episode', type=int, required=True)
    plan.add_argument('--factor', choices=FACTORS, required=True)
    plan.add_argument('--python', default='/home/exaFLOPs/miniforge3/envs/v2d/bin/python')
    for name in ['run', 'register']:
        p = sub.add_parser(name); p.add_argument('--id', required=True)
    sub.add_parser('status')
    args = parser.parse_args();registry = read_registry(args.registry)
    if args.action == 'plan':
        for value in FACTORS[args.factor]:
            config = {**BASE, args.factor: value}
            # Keep the comparison one-factor: static and moving-camera runs
            # both omit gravity rather than changing two factors at once.
            if args.factor == 'camera':
                config['gravity'] = 'off'
            identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
            run_id = f'ep{args.episode:06d}_{identity}'
            output = REPO / '.ablation_lab/runs' / run_id
            record = {'id': run_id, 'episode': args.episode, 'config': config, 'output_dir': str(output), 'status': 'planned'}
            record['command'] = make_command(args.episode, config, output, args.dataset, args.python)
            if not any(r['id'] == run_id for r in registry['runs']):
                registry['runs'].append(record)
            print(run_id, shlex.join(record['command']), sep='\n')
        save(registry, args.registry);return
    if args.action == 'status':
        print(json.dumps(registry, indent=2));return
    record = next(r for r in registry['runs'] if r['id'] == args.id)
    if args.action == 'run':
        out = Path(record['output_dir']);out.mkdir(parents=True, exist_ok=True)
        expected = out / 'experiment.json'
        if expected.exists() and json.loads(expected.read_text())['config'] != record['config']:
            raise ValueError('Refusing to reuse outputs from a different experiment')
        expected.write_text(json.dumps(record, indent=2))
        record['status'] = 'running';record['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        record['provenance'] = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                                'execution': 'fresh pipeline command; stage-level resume may reuse only this experiment directory'}
        save(registry, args.registry)
        try:
            with (out / 'run.log').open('a') as log:
                subprocess.run(record['command'], cwd=REPO / 'reconstruction', stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_result(record)
        except Exception as exc:
            record['status'] = 'failed';record['error'] = str(exc);save(registry,args.registry);raise
    register_scene(record, args.site.resolve(), args.dataset.resolve())
    record['status'] = 'completed';save(registry,args.registry)
    print('Registered:',record['id'])


if __name__ == '__main__':
    main()
