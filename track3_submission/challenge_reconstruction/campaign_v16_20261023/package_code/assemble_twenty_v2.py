#!/usr/bin/env python3
"""Assemble a complete twenty-episode Track 3 package from twenty freshly executed common components (evaluator build v2).

Unlike assemble_twenty_swap.py this does not start from an older package (whose rows carry the previous build's hashes);
every episode comes from a component executed under the given build manifest, so all 9,120 rows share one evaluator hash
pair and commit URL. Postprocessing only: no evaluator change, no Kaggle call, nothing uploaded.
usage: assemble_twenty_v2.py --build-manifest M --common-evaluator E --component EP=DIR x20 --portable EP=TAR x20
       [--video EP=PATH ...] --output DIR --reason TEXT
"""
from __future__ import annotations
import argparse, importlib.util, json, shutil, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
BASE_CODE = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code')
_spec = importlib.util.spec_from_file_location('assemble_base', BASE_CODE / 'assemble_twenty_best_effort.py')
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
sha, read, write, module_at, entries, validate_kit, validate_portable, run_logged = (
    base.sha, base.read, base.write, base.module_at, base.entries, base.validate_kit, base.validate_portable, base.run_logged)
ALL = base.ALL_EPISODES


def validate_component(args, common, build, episode, component, portable):
    name = f'episode_{episode:06d}'; record = read(component / 'completed.json')
    if (record['episode'] != name or not record.get('fresh_backend_executed') or record.get('backend_kind') != 'frozen_mujoco_rbf' or
            record.get('build_manifest_sha256') != sha(args.build_manifest) or
            any(record.get(k) != build[k] for k in ('eval_code_sha256', 'eval_script_sha256', 'code_commit_url')) or
            record.get('exact_previous_physical_states_actions') is not True):
        raise ValueError('Component did not execute under the given build: ' + name)
    for key in ('checkpoint', 'actions', 'backend_export', 'trajectory'):
        if sha(record[key]['path']) != record[key]['sha256']: raise ValueError('Component artifact changed: ' + name + '/' + key)
    for mesh in record['meshes'].values():
        if sha(mesh['path']) != mesh['sha256']: raise ValueError('Component mesh changed: ' + name)
    trajectory = component / 'results' / (name + '.parquet')
    if sha(trajectory) != record['trajectory']['sha256']: raise ValueError('Trajectory location differs from producing report')
    validation = common.official_validate(trajectory, args.kit, episode, record['checkpoint']['sha256'])
    if any(validation['provenance'][k] != build[k] for k in ('eval_script_sha256', 'eval_code_sha256')):
        raise ValueError('Trajectory metadata differs from producing build')
    mesh_validation = common.validate_meshes(component / 'meshes', args.kit, episode, validation['object_names'])
    return dict(record=record, validation=validation, mesh_validation=mesh_validation, portable=validate_portable(portable, episode, record))


def run(args):
    common = module_at('frozen_common_evaluator_v2', args.common_evaluator); build = common.verify_build(args.build_manifest); kit = validate_kit(args.kit)
    components, portables, videos = entries(args.component), entries(args.portable), entries(args.video)
    if sorted(components) != ALL or sorted(portables) != ALL or not set(videos) <= set(ALL): raise ValueError('Supply --component and --portable for all twenty episodes')
    records = {e: validate_component(args, common, build, e, components[e], portables[e]) for e in ALL}
    output = args.output.resolve(); partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists(): raise FileExistsError('Preserve existing output: ' + str(output))
    for d in ('results', 'meshes', 'evidence', 'robot_actions', 'portable_archives', 'videos', 'preparation'): (partial / d).mkdir(parents=True)
    rows = []
    for e in ALL:
        name = f'episode_{e:06d}'; data, report, component = records[e], records[e]['record'], components[e]
        for folder in ('results', 'meshes'): shutil.copytree(component / folder, partial / folder, dirs_exist_ok=True)
        evidence = partial / 'evidence' / name; evidence.mkdir()
        shutil.copy2(component / 'completed.json', evidence / 'common_execution.json'); shutil.copy2(component / 'build_manifest.json', evidence / 'build_manifest.json')
        shutil.copy2(report['backend_export']['path'], evidence / 'original_backend_export.parquet')
        for fn in ('portable_validation.json', 'evaluation/result.json'):
            src = component / 'backend' / fn
            if src.is_file(): (evidence / fn).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, evidence / fn)
        prov = Path(report['legacy_component']) / 'provenance'
        if prov.is_dir():
            for f in prov.glob('*.json'): shutil.copy2(f, evidence / f.name)
        action = partial / 'robot_actions' / (name + '.npz'); shutil.copy2(report['actions']['path'], action)
        portable = partial / 'portable_archives' / (name + '_portable.tar.gz'); shutil.copy2(portables[e], portable)
        video_rel = None; src_video = videos.get(e) or (Path(report['legacy_component']) / 'videos' if (Path(report['legacy_component']) / 'videos').is_dir() else None)
        if src_video is not None:
            target = partial / 'videos' / name
            if src_video.is_dir(): shutil.copytree(src_video, target)
            else: target.mkdir(); shutil.copy2(src_video, target / src_video.name)
            video_rel = str(target.relative_to(partial))
        rows.append(dict(episode=name, backend=report['backend_kind'], expected_rows=data['validation']['rows'], checkpoint_sha256=report['checkpoint']['sha256'],
            trajectory_sha256=report['trajectory']['sha256'], common_execution=str(evidence.relative_to(partial) / 'common_execution.json'),
            actions=str(action.relative_to(partial)), actions_sha256=sha(action), portable_archive=str(portable.relative_to(partial)), archive_sha256=sha(portable),
            metrics=report['metrics'], source_pose_proxy_not_official_ground_truth=True, previous_physical_recording_exact=True, video_directory=video_rel,
            best_effort=False, reference_proxy_mp_sr=1., component_source=str(component)))
    shutil.copytree(args.kit, partial / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    commands = partial / 'preparation'; shutil.copy2(__file__, commands / Path(__file__).name)
    pack = [sys.executable, 'tools/pack_submission.py', '--track', 'track_3', '--results', str(partial / 'results'), '--meshes', str(partial / 'meshes'),
            '--code-commit', build['code_commit_url'], '-o', str(partial / 'track3_submission.parquet')]
    run_logged(pack, args.kit, commands / 'pack.log')
    run_logged([sys.executable, 'tools/preflight.py', '--track', 'track_3', '--results', str(partial / 'results'), '--skip-kaggle'], args.kit, commands / 'preflight_offline.log')
    uploader = module_at('official_uploader_validation_only', args.kit / 'submit/v2d_submit.py')
    frame = uploader.validate_submission(partial / 'track3_submission.parquet', 'track_3', read(args.kit / 'config/tracks.json'))
    episodes = frame.row_id.str.extract(r'/e(\d{6})/')[0].astype(int)
    if sorted(episodes.unique().tolist()) != ALL or len(frame) != 9120: raise ValueError('Packed roster differs from the twenty episodes / 9120 rows')
    for row in rows:
        e = int(row['episode'].split('_')[1])
        if frame.loc[episodes == e, 'checkpoint_sha256'].unique().tolist() != [row['checkpoint_sha256']]: raise ValueError('Checkpoint attribution differs: ' + row['episode'])
    for col in ('eval_script_sha256', 'eval_code_sha256', 'code_commit_url'):
        if frame[col].unique().tolist() != [build[col]]: raise ValueError('Packed rows carry mixed ' + col)
    common.verify_build(args.build_manifest)
    manifest = dict(schema='track3_twenty_episode_release_v3_full', created_utc=datetime.now(timezone.utc).isoformat(), available_episodes=ALL, available_episode_count=20,
        pose_row_count=len(frame), episodes=rows, reason=args.reason, evaluator={k: build[k] for k in ('eval_script_sha256', 'eval_code_sha256', 'code_commit_url')},
        build_manifest_sha256=sha(args.build_manifest), official_kit=kit, combined_submission=dict(path='track3_submission.parquet', sha256=sha(partial / 'track3_submission.parquet')),
        preparation=dict(script_sha256=sha(__file__), official_packer_passed=True, official_uploader_local_validation_passed=True, offline_preflight_passed=True),
        best_effort_episodes=[], all_episodes_pass_local_reference_proxy=True, complete_official_roster=True,
        source_pose_proxy_not_official_ground_truth=True, hidden_ground_truth_scores_available=False, uploaded=False)
    write(partial / 'manifest.json', manifest)
    (partial / 'README.md').write_text(f'''# Track 3 complete twenty-episode package ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

All twenty episodes were freshly executed under common evaluator build v2 (manifest sha256 {sha(args.build_manifest)[:12]}...,
commit {build['code_commit_url']}). Reason: {args.reason}
Scores in manifest.json are reference-proxy values against our own reconstruction, not hidden ground truth. Not uploaded.
''')
    inv = {str(p.relative_to(partial)): sha(p) for p in sorted(partial.rglob('*')) if p.is_file() and p != partial / 'SHA256SUMS.json'}
    write(partial / 'SHA256SUMS.json', inv); partial.replace(output)
    print(json.dumps(dict(output=str(output), submission=str(output / 'track3_submission.parquet'), rows=len(frame), uploaded=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-manifest', type=Path, required=True); p.add_argument('--common-evaluator', type=Path, required=True); p.add_argument('--kit', type=Path, default=base.DEFAULT_KIT)
    p.add_argument('--component', action='append', default=[]); p.add_argument('--portable', action='append', default=[]); p.add_argument('--video', action='append', default=[])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--reason', required=True)
    run(p.parse_args())
