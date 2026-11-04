#!/usr/bin/env python3
"""Stage one passed new actor, verify portable replay, and execute frozen build.

Only new output paths are written. The previous thirteen releases and producing
runtime sources are never edited. MP-SR here uses the reconstructed reference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path('/data1/SRL_V2D')
BASE = ROOT / 'challenge_reconstruction/dexagent_remaining7_20261019'
LEGACY = ROOT / 'challenge_reconstruction/dexagent_final_recon_20261017'
COMMON = ROOT / 'challenge_reconstruction/episode37_ilchold_submission_20261019'
EPISODES = [14, 15, 16, 20, 25, 28, 30]


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
    spec.loader.exec_module(module)
    return module


def command(script, *arguments, cwd=None):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, cwd=cwd)


def provenance(args, adapter, output):
    archive_manifest = args.base / 'input_manifest.json'
    archive = read(archive_manifest)
    if sha(archive['archive']) != archive['archive_sha256']:
        raise ValueError('The original user ZIP differs from its extraction manifest')
    source = Path(adapter['source'])
    object_targets = source / 'object_poses.npz'
    supplied = Path(archive['source']) / f'episode_{args.episode:06d}' / 'object_poses.npz'
    if sha(object_targets) != sha(supplied) or sha(object_targets) != adapter['source_object_poses_sha256']:
        raise ValueError('Source object targets changed between user ZIP and robot adapter')
    evidence = adapter.get('new_input_provenance', {})
    hand_source = Path(evidence.get('hawor_source', source))
    hand_report = hand_source / 'hand_report.json'
    if (not hand_report.is_file() or 'hawor' not in json.dumps(read(hand_report)).lower() or
            sha(hand_report) != evidence.get('hawor_hand_report_sha256')):
        raise ValueError('Expected the recorded HaWoR hand report and exact source hash')
    wrappers = [path for path in [Path(__file__).resolve(),
        args.base / 'code/prepare_seven.py', args.base / 'code/refine_and_train.py',
        args.base / 'code/repair_collision_assets.py'] if path.is_file()]
    repairs = []
    for name in ('collision_repair_provenance.json', 'stereo_fit.json',
                 'stereo_fit_left.npz', 'stereo_fit_right.npz',
                 'hawor_decode_report.json', 'hawor_complete.json'):
        candidates = [hand_source / name, source / name]
        for candidate in candidates:
            if candidate.is_file() and all(sha(candidate) != sha(p) for p in repairs):
                repairs.append(candidate)
    record = dict(schema='remaining_seven_source_lineage_v1', episode=args.episode,
        user_archive=dict(path=archive['archive'], sha256=archive['archive_sha256']),
        archive_manifest=dict(path=str(archive_manifest), sha256=sha(archive_manifest)),
        input_manifest=dict(path=str(args.episode_dir / 'input_manifest.json'),
                            sha256=sha(args.episode_dir / 'input_manifest.json')),
        hand_report=dict(path=str(hand_report), sha256=sha(hand_report)),
        hand_source=adapter.get('hand_source'), original_object_targets_byte_identical=True,
        object_poses_sha256=sha(object_targets), reconstructed_reference_sha256=sha(source / 'reference.parquet'),
        wrappers={str(path): sha(path) for path in wrappers},
        input_repair_evidence={str(path): sha(path) for path in repairs},
        source_pose_proxy_not_official_ground_truth=True, uploaded=False)
    write(output, record)
    return [archive_manifest, hand_report, output, *wrappers, *repairs]


def portable_archive(bundle, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        expected = read(Path(str(destination) + '.json'))
        if sha(destination) != expected['sha256']:
            raise ValueError('Existing portable archive changed')
        return expected
    temporary = Path(str(destination) + '.partial')
    if temporary.exists():
        raise FileExistsError('Preserve incomplete archive for inspection: ' + str(temporary))
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as archive:
        archive.add(bundle, arcname=bundle.parent.name + '_portable')
    temporary.replace(destination)
    result = dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination))
    write(Path(str(destination) + '.json'), result)
    return result


def run(args):
    if args.episode not in EPISODES:
        raise ValueError('Only the seven newly supplied episodes are in scope')
    args.base = args.base.resolve()
    name = f'episode_{args.episode:06d}'
    actor = args.actor_report or args.base / 'refinement' / name / 'completed.json'
    selection = read(actor) if actor.is_file() else {}
    for key in ('checkpoint', 'training_report', 'evaluation', 'episode_dir'):
        value = getattr(args, key) or selection.get(key)
        if value is None:
            raise ValueError('Missing actual successful actor input: ' + key)
        setattr(args, key, Path(value).resolve(strict=True))
    if args.evaluation.is_file():
        args.evaluation = args.evaluation.parent
    report = read(args.evaluation / 'result.json')
    if (report.get('episode') != name or not report.get('learned_policy') or
            not report.get('reference_proxy_success') or
            report.get('policy_checkpoint_sha256') != sha(args.checkpoint) or
            report.get('metrics', {}).get('reference_proxy_mp_sr') != 1. or
            not report.get('execution', {}).get('physics_verified') or
            not report['execution'].get('independent_control_replay', {}).get('ok')):
        raise ValueError('Only a successful actual learned actor with independent physics replay may be staged')
    code = LEGACY / 'code'
    common = module_at('frozen_common_staging_validation', COMMON / 'package_code/common_evaluator.py')
    build = common.verify_build(COMMON / 'build_manifest.json')
    # The legacy development tree has extra search/staging scripts. Its fifteen
    # runtime modules and the complete executed DexAgent tree must match exactly.
    for entry in build['sources']:
        label, separator, relative = entry['key'].partition('/')
        if label not in ('legacy_code', 'legacy_dexagent'):
            continue
        root = code if label == 'legacy_code' else ROOT / 'third_party/DexAgent'
        if sha(root / relative) != entry['sha256']:
            raise ValueError('Legacy runtime source changed: ' + entry['key'])
    component = args.base / 'components' / name
    completed = component / 'staged.json'
    if completed.exists():
        saved = read(completed)
        if saved['checkpoint_sha256'] != sha(args.checkpoint):
            raise ValueError('Existing completed staging selected another checkpoint')
        common_record = read(args.base / 'common_components' / name / 'completed.json')
        if (sha(common_record['trajectory']['path']) != common_record['trajectory']['sha256'] or
                sha(saved['archive']['path']) != saved['archive']['sha256']):
            raise ValueError('Previously completed staging artifacts changed')
        print(json.dumps(saved, indent=2))
        return
    component.mkdir(parents=True, exist_ok=True)
    adapter = read(args.episode_dir / 'input_manifest.json')
    lineage = provenance(args, adapter, component / 'provenance/source_lineage.json')
    if actor.is_file():
        lineage.append(actor)
    lineage.extend(args.lineage_file)
    # Existing staging only accepts a run in its refinement_20261018 namespace.
    # Change its directory variable, not source bytes or physical behavior, and
    # perform an actual replay in this new experiment's own namespace.
    fresh = args.base / 'refinement_20261018' / name / 'evaluation'
    if not (fresh / 'shared_evaluator_validation.json').is_file():
        command(code / 'reevaluate_component.py', '--baseline-report', args.evaluation / 'result.json',
                '--output', fresh, '--episode-dir', args.episode_dir)
    if not read(fresh / 'shared_evaluator_validation.json')['all_physical_states_actions_exact']:
        raise ValueError('New staging replay differs from the selected actual actor recording')
    sys.path[:0] = [str(code), str(ROOT / 'third_party/DexAgent')]
    import stage_policy_component
    import prepare_shared_component
    stage_policy_component.B = args.base
    stage_policy_component.WRITER = Path(build['source_roots']['schema_writer'])
    stage_args = SimpleNamespace(run=fresh, checkpoint=args.checkpoint,
        training_report=args.training_report, base_training_report=args.base_training_report,
        episode_dir=args.episode_dir, output=component, quality_dir=args.quality_dir,
        lineage_file=lineage)
    if not (component / 'portable/bundle_manifest.json').is_file():
        prepare_shared_component.run(stage_args)
    for label, directory in [('legacy_code', component / 'portable/code'),
                             ('legacy_dexagent', component / 'portable/third_party/DexAgent')]:
        common.verify_equivalent_tree(build, label, directory)
    validation = component / 'portable_validation'
    if not (validation / 'portable_validation.json').is_file():
        command(component / 'portable/code/reproduce_policy_component.py', 'replay',
                '--bundle', component / 'portable', '--output', validation, cwd='/tmp')
    checked = read(validation / 'portable_validation.json')
    if not all(checked[key] for key in ['all_physical_states_actions_exact',
            'compiled_model_arrays_options_exact', 'original_weights_unchanged',
            'source_object_targets_unchanged']) or checked['reference_proxy_mp_sr'] != 1.:
        raise ValueError('Portable physical/controller validation did not pass')
    common_output = args.base / 'common_components' / name
    if not (common_output / 'completed.json').is_file():
        command(COMMON / 'package_code/common_evaluator.py', 'evaluate', '--backend', 'legacy',
                '--episode', args.episode, '--legacy-component', component,
                '--build-manifest', COMMON / 'build_manifest.json', '--kit', stage_policy_component.KIT,
                '--output', common_output, cwd='/tmp')
    common_record = read(common_output / 'completed.json')
    if (common_record['checkpoint']['sha256'] != sha(args.checkpoint) or
            common_record['eval_code_sha256'] != build['eval_code_sha256'] or
            not common_record['exact_previous_physical_states_actions']):
        raise ValueError('Common evaluator did not reproduce the selected episode under this build')
    # Add supporting evidence outside the frozen runtime source trees. Its
    # actual policy/physics inventory remains unchanged and independently checked.
    evidence = component / 'portable/release_evidence'
    evidence.mkdir(exist_ok=True)
    for source, target in [(component / 'readiness.json', 'readiness.json'),
            (component / 'provenance/source_lineage.json', 'source_lineage.json'),
            (validation / 'portable_validation.json', 'portable_validation.json'),
            (common_output / 'completed.json', 'common_execution.json')]:
        shutil.copy2(source, evidence / target)
    action = evidence / 'actual_robot_actions.npz'
    if not action.exists():
        shutil.copy2(common_record['actions']['path'], action)
    if sha(action) != common_record['actions']['sha256']:
        raise ValueError('Copied action evidence differs from the actual common replay')
    if args.video:
        video_dir = component / 'videos'
        video_dir.mkdir(exist_ok=True)
        if args.video.is_dir():
            shutil.copytree(args.video, video_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(args.video, video_dir / args.video.name)
    archive = portable_archive(component / 'portable',
        args.base / 'portable_archives' / (name + '_portable.tar.gz'))
    common.verify_build(COMMON / 'build_manifest.json')
    result = dict(episode=name, checkpoint_sha256=sha(args.checkpoint), component=str(component),
        common_component=str(common_output), common_build_manifest_sha256=sha(COMMON / 'build_manifest.json'),
        actual_robot_actions_sha256=common_record['actions']['sha256'], archive=archive,
        portable_physical_states_actions_exact=True, source_targets_unchanged=True,
        reference_proxy_mp_sr=1., source_pose_proxy_not_official_ground_truth=True,
        video_directory=str(component / 'videos') if (component / 'videos').is_dir() else None,
        complete_twenty_episode_submission_ready=False, uploaded=False)
    write(completed, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--actor-report', type=Path)
    for key in ('checkpoint', 'training-report', 'evaluation', 'episode-dir',
                'base-training-report', 'quality-dir', 'video'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--lineage-file', type=Path, action='append', default=[])
    run(parser.parse_args())
