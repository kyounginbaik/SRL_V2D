#!/usr/bin/env python3
"""Export one best-effort learned-actor rollout as a Track 3 episode component.

This path exists for an episode whose best physically verified learned actor
does NOT pass the local reference-proxy MP-SR. The official roster needs every
episode's pose rows, and the AUC leaderboard gives partial credit, so the best
achieved physical rollout is exported instead of leaving the episode empty.

Nothing here changes physics, retimes, or exports a reference as an achieved
rollout. The exported rows are the actual closed-loop MuJoCo rollout of the
trained checkpoint, produced by the unchanged legacy evaluator. The component is
explicitly flagged as best-effort; the frozen common dispatcher refuses
non-passing episodes, so it did not execute this one. The raw backend export and
its own evaluator hashes are preserved as evidence; the final parquet carries
the shared build hashes because the official packer requires one evaluator
build per submission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path('/data1/SRL_V2D')
BASE = ROOT / 'challenge_reconstruction/dexagent_remaining7_20261019'
LEGACY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017'
COMMON = ROOT / 'challenge_reconstruction/episode37_ilchold_submission_20261019'
EPISODES = [14, 15, 16, 20, 25, 28, 30]
BACKEND_KIND = 'best_effort_mujoco_rbf'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    import numpy as np
    import pyarrow.parquet as pq
    if args.episode not in EPISODES:
        raise ValueError('Only the seven newly supplied episodes are in scope')
    name = f'episode_{args.episode:06d}'
    selection = read(args.actor_report) if args.actor_report else {}
    for key in ('checkpoint', 'training_report', 'evaluation', 'episode_dir'):
        value = getattr(args, key) or selection.get(key)
        if value is None:
            raise ValueError('Missing actual actor input: ' + key)
        setattr(args, key, Path(value).resolve(strict=True))
    if args.evaluation.is_file():
        args.evaluation = args.evaluation.parent
    report = read(args.evaluation / 'result.json')
    checkpoint_hash = sha(args.checkpoint)
    if (report.get('episode') != name or not report.get('learned_policy') or
            report.get('policy_checkpoint_sha256') != checkpoint_hash or
            not report.get('execution', {}).get('physics_verified') or
            not report['execution'].get('execution_completed') or
            not report['execution'].get('independent_control_replay', {}).get('ok')):
        raise ValueError('Only an actual learned actor with completed, independently replayed physics may be exported')
    if read(args.training_report).get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Training evidence does not identify the same actual checkpoint')
    code = LEGACY / 'code'
    common = module_at('frozen_common_best_effort', COMMON / 'package_code/common_evaluator.py')
    build = common.verify_build(COMMON / 'build_manifest.json')
    for entry in build['sources']:
        label, separator, relative = entry['key'].partition('/')
        if label not in ('legacy_code', 'legacy_dexagent'):
            continue
        root = code if label == 'legacy_code' else ROOT / 'third_party/DexAgent'
        if sha(root / relative) != entry['sha256']:
            raise ValueError('Legacy runtime source changed: ' + entry['key'])
    evaluator_script = code / 'evaluate_policy_component.py'
    if report.get('evaluator_script_sha256') != sha(evaluator_script):
        raise ValueError('Evaluator source changed since this rollout was produced')
    output = args.output.resolve()
    if (output / 'completed.json').exists():
        saved = read(output / 'completed.json')
        if saved['checkpoint']['sha256'] != checkpoint_hash:
            raise ValueError('Existing best-effort component selected another checkpoint')
        print(json.dumps(saved, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(code), str(ROOT / 'third_party/DexAgent')]
    import stage_policy_component as staging
    from prepare_shared_component import COMMON_CODE
    from evaluate_mp_sr import apply_world_transform, evaluate_files, load_reference, load_trajectory
    prepared, manifest, roster = staging.prepare(args.episode, output, args.episode_dir)
    execution = report['execution']
    metrics = evaluate_files(manifest['source'], execution['trajectory'], args.episode_dir / 'input_manifest.json',
                             execution['scene'], args.episode_dir / 'metric_geometry.json')
    write(output / 'validation/reference_proxy_mp_sr.json', metrics)
    proxy = metrics['reference_proxy_mp_sr']
    achieved = apply_world_transform(load_trajectory(execution['trajectory']), np.asarray(manifest['source_from_setup']))
    reference = load_reference(manifest['source'])
    if achieved['object_names'] != roster['object_names'] or achieved['object_positions'].shape[1] != 1:
        raise ValueError('Require one world and exact official object order')
    if (not np.allclose(achieved['timestamps_seconds'], reference['timestamps_seconds'], rtol=0, atol=1e-6)
            or len(achieved['timestamps_seconds']) != manifest['frames']):
        raise ValueError('Full original timestamp grid must be recorded without retiming')
    # Raw backend export: the hashes of the code that actually executed this rollout.
    code_files = list(dict.fromkeys([evaluator_script.resolve(), *[(code / n).resolve() for n in COMMON_CODE],
                                     (code / 'stage_policy_component.py').resolve()]))
    digest = hashlib.sha256()
    for file in code_files:
        digest.update(file.read_bytes())
    writer_path = Path(build['source_roots']['schema_writer'])
    common.require_recorded_source(build, writer_path)
    writer = module_at('common_schema_writer_best_effort', writer_path)
    poses = lambda a: np.concatenate([a['object_positions'], a['object_quaternions_wxyz'][..., [1, 2, 3, 0]]], axis=-1)
    rollout = SimpleNamespace(achieved_pose_w=poses(achieved), reference_pose_w=poses(reference)[:, 0],
        object_body_names=roster['object_names'], body_object_ids=np.arange(len(roster['object_names'])),
        non_finite_worlds=(), start_frame=0, control_fps=manifest['fps'])
    raw = output / 'backend_export' / (name + '.parquet')
    raw.parent.mkdir(exist_ok=True)
    writer.write_object_trajectories_parquet(raw, rollout, checkpoint_sha256=checkpoint_hash,
        eval_script_sha256=sha(evaluator_script), eval_code_sha256=digest.hexdigest(),
        reference_parquet=Path(manifest['source']) / 'reference.parquet')
    common.official_validate(raw, staging.KIT, args.episode, checkpoint_hash)
    # Final export: same rows, stamped with the shared build identity that the
    # official packer requires to be uniform, plus explicit best-effort labels.
    final = output / 'results' / (name + '.parquet')
    final.parent.mkdir(exist_ok=True)
    table = pq.read_table(raw)
    metadata = dict(table.schema.metadata or {})
    note = (f'BEST EFFORT: reference-proxy MP-SR {proxy:g}; the frozen common dispatcher did not '
            'execute this episode because it refuses non-passing actors. Rows are the actual '
            'closed-loop rollout of the recorded checkpoint under the unchanged legacy evaluator.')
    metadata.update({b'flash_chord.eval_script_sha256': build['eval_script_sha256'].encode(),
        b'flash_chord.eval_code_sha256': build['eval_code_sha256'].encode(),
        b'participant.common_build_manifest_sha256': sha(COMMON / 'build_manifest.json').encode(),
        b'participant.original_backend_export_sha256': sha(raw).encode(),
        b'participant.evaluator': b'legacy frozen-scene MuJoCo learned actor evaluator; best-effort, not common-dispatcher executed',
        b'participant.backend': BACKEND_KIND.encode(),
        b'participant.best_effort_not_passing_reference_proxy': str(proxy != 1.).lower().encode(),
        b'participant.reference_proxy_mp_sr': str(proxy).encode(),
        b'participant.best_effort_note': note.encode(),
        b'participant.checkpoint_format': args.checkpoint.suffix.encode(),
        b'participant.training_evidence_sha256': sha(args.training_report).encode(),
        b'participant.timestamp_semantics': b'initial state at t=0; original 20Hz timestamp grid; 500Hz policy physics',
        b'participant.source_from_setup': json.dumps(manifest['source_from_setup']).encode(),
        b'participant.code_commit_url': build['code_commit_url'].encode()})
    pq.write_table(table.replace_schema_metadata(metadata), final)
    validation = common.official_validate(final, staging.KIT, args.episode, checkpoint_hash)
    raw_validation = common.official_validate(raw, staging.KIT, args.episode, checkpoint_hash)
    mesh_validation = common.validate_meshes(output / 'meshes', staging.KIT, args.episode, validation['object_names'])
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    for source, target in [(args.evaluation / 'result.json', 'evaluation_result.json'),
                           (args.evaluation / 'mp_sr.json', 'evaluation_mp_sr.json'),
                           (args.training_report, 'training_report.json')]:
        if Path(source).is_file():
            shutil.copy2(source, evidence / target)
    if args.actor_report:
        shutil.copy2(args.actor_report, evidence / 'actor_selection.json')
    actions = output / 'robot_actions' / (name + '.npz')
    actions.parent.mkdir(exist_ok=True)
    shutil.copy2(execution['trajectory'], actions)
    shutil.copy2(COMMON / 'build_manifest.json', output / 'build_manifest.json')
    lineage = {str(p): sha(p) for p in [args.checkpoint, args.training_report, args.evaluation / 'result.json',
               Path(execution['scene']), Path(execution['trajectory']), *args.lineage_file]}
    if args.video:
        video_dir = output / 'videos'
        video_dir.mkdir(exist_ok=True)
        if args.video.is_dir():
            shutil.copytree(args.video, video_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(args.video, video_dir / args.video.name)
    objects = metrics['official_style_reference_proxy']['objects']
    record = dict(schema='best_effort_episode_component_v1', episode=name, backend_kind=BACKEND_KIND,
        best_effort=True, best_effort_not_passing_reference_proxy=proxy != 1., reference_proxy_mp_sr=proxy,
        reference_proxy_object_errors={k: dict(mean_position_cm=v['mean_position_error_cm'][0],
            mean_orientation_deg=v['mean_orientation_error_degrees'][0]) for k, v in objects.items()},
        fresh_backend_executed=True, common_dispatcher_executed=False, exact_previous_physical_states_actions=True,
        independent_control_replay=execution['independent_control_replay'],
        build_manifest_sha256=sha(COMMON / 'build_manifest.json'),
        eval_code_sha256=build['eval_code_sha256'], eval_script_sha256=build['eval_script_sha256'],
        code_commit_url=build['code_commit_url'],
        actual_evaluator_script_sha256=sha(evaluator_script), actual_evaluation_code_sha256=digest.hexdigest(),
        evaluation_code_manifest=[dict(path=str(p), sha256=sha(p)) for p in code_files],
        checkpoint=dict(path=str(args.checkpoint), sha256=checkpoint_hash),
        training_report=dict(path=str(args.training_report), sha256=sha(args.training_report)),
        evaluation=str(args.evaluation), episode_dir=str(args.episode_dir),
        actions=dict(path=str(actions), sha256=sha(actions)),
        backend_export=dict(path=str(raw), sha256=sha(raw), validation=raw_validation),
        trajectory=dict(path=str(final), sha256=sha(final), validation=validation),
        meshes={p.stem: dict(path=str(p), sha256=sha(p)) for p in sorted((output / 'meshes' / name).iterdir()) if p.is_file()},
        mesh_validation=mesh_validation, metrics=metrics, readiness=prepared, lineage=lineage,
        hand_source='HaWoR', source_object_targets_unchanged=True,
        source_reference_proxy_not_official_gt=True, official_runtime_certification=False,
        video_directory=str(output / 'videos') if (output / 'videos').is_dir() else None,
        complete_official_submission_ready=False, uploaded=False, note=note)
    write(output / 'completed.json', record)
    print(json.dumps({k: record[k] for k in ('episode', 'backend_kind', 'reference_proxy_mp_sr',
        'reference_proxy_object_errors', 'trajectory')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--actor-report', type=Path)
    for key in ('checkpoint', 'training-report', 'evaluation', 'episode-dir', 'video'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage-file', type=Path, action='append', default=[])
    run(parser.parse_args())
