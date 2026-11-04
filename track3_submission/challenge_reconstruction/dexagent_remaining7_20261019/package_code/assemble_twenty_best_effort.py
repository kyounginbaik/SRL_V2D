#!/usr/bin/env python3
"""Combine preserved thirteen and seven new Track 3 components, allowing best-effort episodes.

This is assemble_twenty.py plus `--best-effort EPISODE=component`: an episode whose
best physically verified learned actor does not pass the local reference-proxy
MP-SR is included as its actual best rollout (built by best_effort_component.py)
instead of being left out of the required twenty-episode roster. Such episodes
are explicitly flagged in the manifest and README. Frozen-build components are
validated exactly as before.

This is postprocessing, not an evaluator. It never changes producing evaluator
hashes, exports a reference as an achieved rollout, or calls a Kaggle API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

OLD_EPISODES = [1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38]
NEW_EPISODES = [14, 15, 16, 20, 25, 28, 30]
ALL_EPISODES = sorted(OLD_EPISODES + NEW_EPISODES)
DEFAULT_BASE = Path('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019')
DEFAULT_OLD = Path('/data1/SRL_V2D/challenge_submissions/track3_mixed_ilchold_20261019/available_13_episodes')
DEFAULT_KIT = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


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


def entries(values):
    result = {}
    for value in values:
        episode, separator, path = value.partition('=')
        if not separator:
            raise ValueError('Expected EPISODE=/absolute/path: ' + value)
        episode = int(episode.removeprefix('episode_'))
        if episode in result or not path:
            raise ValueError('Duplicate episode or missing path: ' + value)
        result[episode] = Path(path).resolve(strict=True)
    return result


def safe_relative(value):
    relative = PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe relative path: ' + value)
    return Path(*relative.parts)


def verify_inventory(directory):
    inventory = read(directory / 'SHA256SUMS.json')
    for relative, expected in inventory.items():
        path = directory / safe_relative(relative)
        if sha(path) != expected:
            raise ValueError('Preserved package checksum mismatch: ' + str(path))
    return inventory


def validate_kit(kit):
    manifest = read(kit / 'KIT_MANIFEST.json')
    for relative, expected in manifest['files'].items():
        if sha(kit / safe_relative(relative)) != expected:
            raise ValueError('Official kit changed: ' + relative)
    config = read(kit / 'config/tracks.json')
    spec = config['tracks']['track_3']
    if (sorted(map(int, spec['scored_episodes'])) != ALL_EPISODES or
            not spec.get('register_object_mesh') or spec.get('submission_format') != 'parquet'):
        raise ValueError('Expected the unchanged full twenty-episode combined Parquet contract')
    return dict(path=str(kit), manifest_sha256=sha(kit / 'KIT_MANIFEST.json'),
                upstream_evaluator=manifest['evaluator'], files_verified=len(manifest['files']))


def validate_portable(path, episode, report):
    """Read archive members without extraction and bind its actual policy bytes."""
    with tarfile.open(path, 'r:*') as archive:
        manifests = [member for member in archive.getmembers()
                     if member.isfile() and member.name.endswith('/bundle_manifest.json')]
        if len(manifests) != 1:
            raise ValueError('Expected one portable bundle manifest: ' + str(path))
        member = manifests[0]
        safe_relative(member.name)
        with archive.extractfile(member) as stream:
            bundle = json.load(stream)
        if (bundle.get('episode_id') != episode or
                bundle.get('original_checkpoint_sha256') != report['checkpoint']['sha256'] or
                bundle.get('source_reference_sha256') != report['source_reference_sha256']):
            raise ValueError('Portable archive does not identify the evaluated episode/policy/reference')
        checkpoint = str(PurePosixPath(member.name).parent / 'original/policy.npz')
        with archive.extractfile(checkpoint) as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != report['checkpoint']['sha256']:
            raise ValueError('Portable archive contains different policy weights')
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                actual_checkpoint_verified=True, source_reference_sha256=bundle['source_reference_sha256'])


def audit_existing(args, common, build):
    old = read(args.original_package / 'manifest.json')
    if old['available_episodes'] != OLD_EPISODES or old['evaluator'] != {
            key: build[key] for key in ('eval_script_sha256', 'eval_code_sha256', 'code_commit_url')}:
        raise ValueError('Existing thirteen components do not use this exact producing build')
    if old['build_manifest_sha256'] != sha(args.build_manifest):
        raise ValueError('Existing build manifest binding differs')
    inventory = verify_inventory(args.original_package)
    validations = {}
    for episode in OLD_EPISODES:
        name = f'episode_{episode:06d}'
        record = read(args.original_package / 'evidence' / name / 'common_execution.json')
        trajectory = args.original_package / 'results' / (name + '.parquet')
        if sha(trajectory) != record['trajectory']['sha256']:
            raise ValueError('Preserved trajectory differs from actual execution: ' + name)
        validation = common.official_validate(trajectory, args.kit, episode, record['checkpoint']['sha256'])
        if any(validation['provenance'][key] != build[key]
               for key in ('eval_script_sha256', 'eval_code_sha256')):
            raise ValueError('Preserved trajectory has different evaluator provenance: ' + name)
        validations[name] = validation
    return old, inventory, validations


def validate_best_effort(args, common, build, episode, component):
    name = f'episode_{episode:06d}'
    record = read(component / 'completed.json')
    if (record['episode'] != name or record.get('backend_kind') != 'best_effort_mujoco_rbf' or
            not record.get('best_effort') or not record.get('fresh_backend_executed') or
            record.get('build_manifest_sha256') != sha(args.build_manifest) or
            any(record.get(key) != build[key] for key in
                ('eval_code_sha256', 'eval_script_sha256', 'code_commit_url')) or
            not record.get('independent_control_replay', {}).get('ok') or
            record.get('exact_previous_physical_states_actions') is not True):
        raise ValueError('Best-effort component is not a flagged actual rollout under this build: ' + name)
    for key in ('checkpoint', 'actions', 'backend_export', 'trajectory', 'training_report'):
        if sha(record[key]['path']) != record[key]['sha256']:
            raise ValueError('Best-effort component artifact changed: ' + name + '/' + key)
    for mesh in record['meshes'].values():
        if sha(mesh['path']) != mesh['sha256']:
            raise ValueError('Best-effort component mesh changed: ' + name)
    trajectory = component / 'results' / (name + '.parquet')
    if sha(trajectory) != record['trajectory']['sha256']:
        raise ValueError('Best-effort trajectory location differs from producing report')
    validation = common.official_validate(trajectory, args.kit, episode, record['checkpoint']['sha256'])
    if any(validation['provenance'][key] != build[key]
           for key in ('eval_script_sha256', 'eval_code_sha256')):
        raise ValueError('Best-effort trajectory metadata differs from shared build identity')
    mesh_validation = common.validate_meshes(component / 'meshes', args.kit, episode, validation['object_names'])
    return dict(record=record, validation=validation, mesh_validation=mesh_validation, portable=None, best_effort=True)


def validate_new(args, common, build, components, portables, best_effort):
    frozen = set(components)
    if (frozen & set(best_effort) or (frozen | set(best_effort)) != set(NEW_EPISODES) or
            set(portables) != frozen):
        raise ValueError('Supply each of the seven episodes once as --component (with --portable) or --best-effort: ' + str(NEW_EPISODES))
    records = {}
    for episode in sorted(best_effort):
        records[episode] = validate_best_effort(args, common, build, episode, best_effort[episode])
    for episode in sorted(frozen):
        name = f'episode_{episode:06d}'
        component = components[episode]
        record = read(component / 'completed.json')
        if (record['episode'] != name or not record.get('fresh_backend_executed') or
                record.get('backend_kind') != 'frozen_mujoco_rbf' or
                record.get('build_manifest_sha256') != sha(args.build_manifest) or
                any(record.get(key) != build[key] for key in
                    ('eval_code_sha256', 'eval_script_sha256', 'code_commit_url')) or
                record.get('exact_previous_physical_states_actions') is not True):
            raise ValueError('New component did not execute under the unchanged frozen build: ' + name)
        for key in ('checkpoint', 'actions', 'backend_export', 'trajectory'):
            if sha(record[key]['path']) != record[key]['sha256']:
                raise ValueError('New component artifact changed: ' + name + '/' + key)
        for mesh in record['meshes'].values():
            if sha(mesh['path']) != mesh['sha256']:
                raise ValueError('New component mesh changed: ' + name)
        trajectory = component / 'results' / (name + '.parquet')
        if sha(trajectory) != record['trajectory']['sha256']:
            raise ValueError('New component trajectory location differs from producing report')
        validation = common.official_validate(trajectory, args.kit, episode, record['checkpoint']['sha256'])
        if any(validation['provenance'][key] != build[key]
               for key in ('eval_script_sha256', 'eval_code_sha256')):
            raise ValueError('New trajectory metadata differs from producing build')
        mesh_validation = common.validate_meshes(component / 'meshes', args.kit, episode, validation['object_names'])
        portable = validate_portable(portables[episode], episode, record)
        records[episode] = dict(record=record, validation=validation,
                                mesh_validation=mesh_validation, portable=portable, best_effort=False)
    return records


def run_logged(command, cwd, log):
    with Path(log).open('w') as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run(args):
    common = module_at('frozen_common_evaluator', args.common_evaluator)
    build = common.verify_build(args.build_manifest)
    kit = validate_kit(args.kit)
    old, original_inventory, old_validation = audit_existing(args, common, build)
    if args.audit_existing_only:
        print(json.dumps(dict(old_episodes_verified=len(old_validation),
            old_files_verified=len(original_inventory), official_kit=kit,
            producing_commit=build['code_commit_url'], uploaded=False), indent=2))
        return
    if args.output is None:
        raise ValueError('--output is required for assembly')
    output = args.output.resolve()
    partial = output.with_name(output.name + '.partial')
    outer_archive = Path(str(output) + '.tar')
    if any(path.exists() for path in (output, partial, outer_archive)):
        raise FileExistsError('Preserve existing output/partial/archive: ' + str(output))
    components, portables, videos = entries(args.component), entries(args.portable), entries(args.video)
    best_effort = entries(args.best_effort)
    if not set(videos).issubset(NEW_EPISODES):
        raise ValueError('--video may only add the new seven episodes')
    records = validate_new(args, common, build, components, portables, best_effort)
    shutil.copytree(args.original_package, partial)
    evidence_root = partial / 'evidence'
    preserved = evidence_root / 'original_thirteen_release'
    preserved.mkdir()
    for filename in ('manifest.json', 'SHA256SUMS.json', 'README.md'):
        shutil.copy2(args.original_package / filename, preserved / filename)
    rows = list(old['episodes'])
    for episode in NEW_EPISODES:
        name = f'episode_{episode:06d}'
        data, report = records[episode], records[episode]['record']
        component = best_effort[episode] if data['best_effort'] else components[episode]
        for folder in ('results', 'meshes'):
            shutil.copytree(component / folder, partial / folder, dirs_exist_ok=True)
        evidence = evidence_root / name
        evidence.mkdir()
        shutil.copy2(component / 'completed.json', evidence / (
            'best_effort_execution.json' if data['best_effort'] else 'common_execution.json'))
        shutil.copy2(component / 'build_manifest.json', evidence / 'build_manifest.json')
        shutil.copy2(report['backend_export']['path'], evidence / 'original_backend_export.parquet')
        if data['best_effort']:
            shutil.copytree(component / 'evidence', evidence / 'best_effort_evidence')
            for filename in ('readiness.json', 'validation/reference_proxy_mp_sr.json'):
                if (component / filename).is_file():
                    target = evidence / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(component / filename, target)
        for filename in ('portable_validation.json', 'evaluation/result.json'):
            source = component / 'backend' / filename
            if source.is_file():
                target = evidence / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        action = partial / 'robot_actions' / (name + '.npz')
        shutil.copy2(report['actions']['path'], action)
        portable = None
        if not data['best_effort']:
            portable = partial / 'portable_archives' / (name + '_portable.tar.gz')
            shutil.copy2(portables[episode], portable)
        source_video = videos.get(episode)
        if source_video is None:
            candidate = Path(report.get('legacy_component') or component) / 'videos'
            source_video = candidate if candidate.is_dir() else None
        video_relative = None
        if source_video is not None:
            target = partial / 'videos' / name
            if source_video.is_dir():
                shutil.copytree(source_video, target)
            else:
                target.mkdir()
                shutil.copy2(source_video, target / source_video.name)
            video_relative = str(target.relative_to(partial))
        rows.append(dict(episode=name, backend=report['backend_kind'],
            expected_rows=data['validation']['rows'], checkpoint_sha256=report['checkpoint']['sha256'],
            trajectory_sha256=report['trajectory']['sha256'],
            common_execution=str(evidence.relative_to(partial) / 'common_execution.json'),
            actions=str(action.relative_to(partial)), actions_sha256=sha(action),
            portable_archive=str(portable.relative_to(partial)) if portable else None,
            archive_sha256=sha(portable) if portable else None,
            metrics=report['metrics'], source_pose_proxy_not_official_ground_truth=True,
            previous_physical_recording_exact=True, video_directory=video_relative,
            best_effort=data['best_effort'],
            reference_proxy_mp_sr=report.get('reference_proxy_mp_sr', 1. if not data['best_effort'] else None),
            best_effort_object_errors=report.get('reference_proxy_object_errors') if data['best_effort'] else None))
    rows.sort(key=lambda row: row['episode'])
    # Include the unchanged official tools; no credentials or account files are copied.
    shutil.copytree(args.kit, partial / 'submission_kit',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    commands = partial / 'preparation'
    commands.mkdir()
    shutil.copy2(__file__, commands / Path(__file__).name)
    pack = [sys.executable, 'tools/pack_submission.py', '--track', 'track_3',
            '--results', str(partial / 'results'), '--meshes', str(partial / 'meshes'),
            '--code-commit', build['code_commit_url'], '-o', str(partial / 'track3_submission.parquet')]
    run_logged(pack, args.kit, commands / 'pack.log')
    preflight = [sys.executable, 'tools/preflight.py', '--track', 'track_3',
                 '--results', str(partial / 'results'), '--skip-kaggle']
    run_logged(preflight, args.kit, commands / 'preflight_offline.log')
    uploader = module_at('official_uploader_validation_only', args.kit / 'submit/v2d_submit.py')
    frame = uploader.validate_submission(partial / 'track3_submission.parquet', 'track_3',
                                        read(args.kit / 'config/tracks.json'))
    episodes = frame.row_id.str.extract(r'/e(\d{6})/')[0].astype(int)
    if sorted(episodes.unique().tolist()) != ALL_EPISODES or len(frame) != 9120:
        raise ValueError('Packed full public roster differs from expected twenty episodes / 9120 rows')
    for row in rows:
        episode = int(row['episode'].split('_')[1])
        values = frame.loc[episodes == episode, 'checkpoint_sha256'].unique().tolist()
        if values != [row['checkpoint_sha256']]:
            raise ValueError('Packed episode checkpoint attribution differs: ' + row['episode'])
    # Ensure the previous thirteen data/action/video/code files remained byte-identical.
    replacements = {'manifest.json', 'README.md', 'SHA256SUMS.json'}
    for relative, expected in original_inventory.items():
        if relative not in replacements and sha(partial / relative) != expected:
            raise ValueError('Assembly altered an existing thirteen-episode artifact: ' + relative)
    common.verify_build(args.build_manifest)
    manifest = dict(schema='track3_twenty_episode_release_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        available_episodes=ALL_EPISODES, available_episode_count=20, pose_row_count=len(frame), episodes=rows,
        preserved_episodes=OLD_EPISODES, added_episodes=NEW_EPISODES,
        original_package=str(args.original_package), original_manifest_sha256=sha(args.original_package / 'manifest.json'),
        preserved_thirteen_artifacts_byte_identical=True, evaluator=old['evaluator'],
        build_manifest_sha256=sha(args.build_manifest), official_kit=kit,
        combined_submission=dict(path='track3_submission.parquet', sha256=sha(partial / 'track3_submission.parquet')),
        preparation=dict(script_sha256=sha(__file__), code_commit_url=args.preparation_commit_url,
                         evaluator_commit_is_unchanged=True, official_packer_passed=True,
                         official_uploader_local_validation_passed=True, offline_preflight_passed=True),
        best_effort_episodes=sorted(best_effort), all_episodes_pass_local_reference_proxy=not best_effort,
        complete_official_roster=True, complete_payload_locally_validated=True,
        source_pose_proxy_not_official_ground_truth=True, hidden_ground_truth_scores_available=False,
        official_runtime_certification=False, kaggle_account_readiness_checked=False, uploaded=False)
    write(partial / 'manifest.json', manifest)
    best_effort_text = ''
    if best_effort:
        lines = []
        for episode in sorted(best_effort):
            errors = records[episode]['record'].get('reference_proxy_object_errors', {})
            detail = '; '.join(f"{k}: {v['mean_position_cm']:.2f} cm / {v['mean_orientation_deg']:.1f} deg"
                               for k, v in errors.items())
            lines.append(f'- episode_{episode:06d}: local reference-proxy MP-SR '
                         f"{records[episode]['record']['reference_proxy_mp_sr']:g} ({detail})")
        best_effort_text = '''
## Best-effort episodes

The following episode(s) did not reach the local reference-proxy MP-SR threshold
(every object below 3 cm mean position and 30 deg mean orientation error) with any
physically verified learned actor before packaging. Their rows are the actual
closed-loop rollout of the best trained checkpoint under the unchanged legacy
evaluator, exported by `best_effort_component.py`, so the required roster is
complete and the AUC/RPE/MPPE leaderboards receive the achieved poses. They are
NOT claimed as passing: the frozen common dispatcher refuses non-passing actors
and did not execute them, no portable archive is included for them, and their
evidence is under `evidence/episode_XXXXXX/best_effort_execution.json`.

''' + '\n'.join(lines) + '\n'
    (partial / 'README.md').write_text('''# Complete twenty-episode Track 3 payload

`track3_submission.parquet` contains all 9,120 public pose rows and the reconstructed
meshes for all twenty required episodes. It was built by the unchanged official
packer and passed the official uploader's local validation and offline preflight.
No Kaggle API was called; competition entry and account readiness are not checked.

The previous thirteen achieved trajectories, action recordings, checkpoints,
videos and evaluation code are preserved byte-for-byte. Episode 37 remains the
selected ilchold policy/reference with its separately rendered fresh replay.
Episodes 14, 15, 16, 20, 25, 28 and 30 are freshly executed DexAgent components
under that same frozen evaluator build. The evaluator commit is distinct from
later preparation code. Original evidence and per-episode portable archives are
retained, along with the full producing code manifest.

`robot_actions/` stores actual backend-native actions and physical states. The
DexAgent and ilchold action representations/rates differ; follow each archive's
README instead of interpreting one as the other's command space.

Local MP-SR results compare against reconstructed source references. Confidential
official poses/meshes are unavailable, so these are not official leaderboard
scores and do not guarantee competition success. Meshes and translations use
metres and the submitted body-frame convention; pose quaternions use xyzw.

The included `submission_kit/` contains the official validation/upload tools.
To upload after checking the account and competition rules, run from that kit:

    python tools/preflight.py --track track_3 --results ../results
    python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "Track 3 full twenty episodes" --wait

The uploader sends the combined file to the five pose competitions and extracts
the mesh payload for CD-O. This package has not been uploaded.
''' + best_effort_text)
    inventory = {str(path.relative_to(partial)): sha(path) for path in sorted(partial.rglob('*'))
                 if path.is_file() and path != partial / 'SHA256SUMS.json'}
    write(partial / 'SHA256SUMS.json', inventory)
    partial.replace(output)
    if args.archive:
        temporary_archive = Path(str(outer_archive) + '.partial')
        with tarfile.open(temporary_archive, 'w') as archive:
            archive.add(output, arcname=output.name)
        temporary_archive.replace(outer_archive)
        write(Path(str(outer_archive) + '.json'), dict(path=str(outer_archive),
              bytes=outer_archive.stat().st_size, sha256=sha(outer_archive)))
    print(json.dumps(dict(output=str(output), submission=str(output / 'track3_submission.parquet'),
                         episodes=20, rows=len(frame), uploaded=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-package', type=Path, default=DEFAULT_OLD)
    parser.add_argument('--build-manifest', type=Path, default=DEFAULT_BASE / 'build_manifest.json')
    parser.add_argument('--common-evaluator', type=Path, default=DEFAULT_BASE / 'package_code/common_evaluator.py')
    parser.add_argument('--kit', type=Path, default=DEFAULT_KIT)
    parser.add_argument('--component', action='append', default=[], help='EPISODE=fresh common component directory')
    parser.add_argument('--portable', action='append', default=[], help='EPISODE=portable archive')
    parser.add_argument('--best-effort', action='append', default=[], help='EPISODE=best-effort component directory (no portable archive)')
    parser.add_argument('--video', action='append', default=[], help='EPISODE=reviewed video file or directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preparation-commit-url', default=None)
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--audit-existing-only', action='store_true')
    run(parser.parse_args())
