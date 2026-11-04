#!/usr/bin/env python3
"""Replace selected episodes of an existing complete twenty-episode Track 3 package with fresh frozen-build components.

Everything not replaced stays byte-identical. The replaced episodes' previous evidence is preserved under
evidence/replaced_<date>/. The combined Parquet is re-packed with the unchanged official packer and re-validated.
This is postprocessing, not an evaluator; it never changes producing evaluator hashes and never calls Kaggle.
"""
from __future__ import annotations
import argparse, importlib.util, json, shutil, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('assemble_base', HERE / 'assemble_twenty_best_effort.py')
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
sha, read, write, module_at, entries, safe_relative, verify_inventory, validate_kit, validate_portable, run_logged = (
    base.sha, base.read, base.write, base.module_at, base.entries, base.safe_relative, base.verify_inventory, base.validate_kit, base.validate_portable, base.run_logged)
ALL = base.ALL_EPISODES


def validate_component(args, common, build, episode, component, portable):
    name = f'episode_{episode:06d}'; record = read(component / 'completed.json')
    if (record['episode'] != name or not record.get('fresh_backend_executed') or record.get('backend_kind') != 'frozen_mujoco_rbf' or
            record.get('build_manifest_sha256') != sha(args.build_manifest) or
            any(record.get(k) != build[k] for k in ('eval_code_sha256', 'eval_script_sha256', 'code_commit_url')) or
            record.get('exact_previous_physical_states_actions') is not True):
        raise ValueError('Component did not execute under the unchanged frozen build: ' + name)
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
    common = module_at('frozen_common_evaluator', args.common_evaluator); build = common.verify_build(args.build_manifest); kit = validate_kit(args.kit)
    old = read(args.original_package / 'manifest.json')
    if sorted(old['available_episodes']) != ALL or old['evaluator'] != {k: build[k] for k in ('eval_script_sha256', 'eval_code_sha256', 'code_commit_url')}:
        raise ValueError('Original package is not a complete twenty-episode package under this build')
    inventory = verify_inventory(args.original_package)
    components, portables, videos = entries(args.component), entries(args.portable), entries(args.video)
    swap = sorted(components)
    if set(portables) != set(swap) or not set(videos) <= set(swap) or not set(swap) <= set(ALL): raise ValueError('Supply --component and --portable for each swapped episode')
    records = {e: validate_component(args, common, build, e, components[e], portables[e]) for e in swap}
    output = args.output.resolve(); partial = output.with_name(output.name + '.partial'); outer = Path(str(output) + '.tar')
    if any(p.exists() for p in (output, partial, outer)): raise FileExistsError('Preserve existing output: ' + str(output))
    shutil.copytree(args.original_package, partial, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d'); replaced_root = partial / 'evidence' / f'replaced_{stamp}'
    if replaced_root.exists(): replaced_root = partial / 'evidence' / f"replaced_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}"
    replaced_root.mkdir()
    for fn in ('manifest.json', 'SHA256SUMS.json', 'README.md'): shutil.copy2(args.original_package / fn, replaced_root / fn)
    rows = [r for r in old['episodes'] if int(r['episode'].split('_')[1]) not in swap]
    for e in swap:
        name = f'episode_{e:06d}'; old_row = next(r for r in old['episodes'] if r['episode'] == name)
        keep = replaced_root / name; keep.mkdir(); write(keep / 'previous_manifest_row.json', old_row)
        for rel in (f'evidence/{name}', f'videos/{name}', f'meshes/{name}'):
            if (partial / rel).exists(): shutil.move(str(partial / rel), str(keep / Path(rel).name) if rel.startswith('evidence') else str(keep / ('previous_' + Path(rel).parent.name)))
        for rel in (f'results/{name}.parquet', f'robot_actions/{name}.npz', f'portable_archives/{name}_portable.tar.gz'):
            if (partial / rel).exists(): shutil.move(str(partial / rel), str(keep / Path(rel).name))
        data, report, component = records[e], records[e]['record'], components[e]
        for folder in ('results', 'meshes'): shutil.copytree(component / folder, partial / folder, dirs_exist_ok=True)
        evidence = partial / 'evidence' / name; evidence.mkdir()
        shutil.copy2(component / 'completed.json', evidence / 'common_execution.json'); shutil.copy2(component / 'build_manifest.json', evidence / 'build_manifest.json')
        shutil.copy2(report['backend_export']['path'], evidence / 'original_backend_export.parquet')
        for fn in ('portable_validation.json', 'evaluation/result.json'):
            src = component / 'backend' / fn
            if src.is_file(): (evidence / fn).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, evidence / fn)
        lineage = Path(report['legacy_component']) / 'provenance/pot_carry_lineage.json'
        if lineage.is_file(): shutil.copy2(lineage, evidence / 'pot_carry_lineage.json')
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
            best_effort=False, reference_proxy_mp_sr=1., replaced_previous_component=True, replacement_reason=args.reason))
    rows.sort(key=lambda r: r['episode'])
    commands = partial / 'preparation'; commands.mkdir(exist_ok=True); shutil.copy2(__file__, commands / Path(__file__).name)
    (partial / 'track3_submission.parquet').unlink(); (partial / 'track3_submission.cd-o.parquet').unlink(missing_ok=True)
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
    # rows of the kept episodes must be identical to the previous combined file
    import pandas as pd
    before = pd.read_parquet(args.original_package / 'track3_submission.parquet'); after = frame
    keep_mask_b = ~before.row_id.str.extract(r'/e(\d{6})/')[0].astype(int).isin(swap); keep_mask_a = ~episodes.isin(swap)
    cols = ['row_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'checkpoint_sha256', 'object_mesh_b64']
    b = before.loc[keep_mask_b, cols].sort_values('row_id').reset_index(drop=True); a_ = after.loc[keep_mask_a, cols].sort_values('row_id').reset_index(drop=True)
    if not b.equals(a_): raise ValueError('Kept episodes changed in the re-packed combined file')
    replacements = {'manifest.json', 'README.md', 'SHA256SUMS.json', 'track3_submission.parquet', 'track3_submission.cd-o.parquet'}
    swapped_prefixes = tuple(p for e in swap for p in (f'evidence/episode_{e:06d}/', f'videos/episode_{e:06d}/', f'meshes/episode_{e:06d}/', f'results/episode_{e:06d}.parquet', f'robot_actions/episode_{e:06d}.npz', f'portable_archives/episode_{e:06d}_portable.tar.gz'))
    for rel, expected in inventory.items():
        if rel in replacements or rel.startswith(swapped_prefixes) or rel.startswith('preparation/'): continue
        if sha(partial / rel) != expected: raise ValueError('Assembly altered a kept artifact: ' + rel)
    common.verify_build(args.build_manifest)
    manifest = dict(schema='track3_twenty_episode_release_v2_swap', created_utc=datetime.now(timezone.utc).isoformat(), available_episodes=ALL, available_episode_count=20,
        pose_row_count=len(frame), episodes=rows, replaced_episodes=swap, replacement_reason=args.reason, original_package=str(args.original_package),
        original_manifest_sha256=sha(args.original_package / 'manifest.json'), kept_artifacts_byte_identical=True, evaluator=old['evaluator'],
        build_manifest_sha256=sha(args.build_manifest), official_kit=kit, combined_submission=dict(path='track3_submission.parquet', sha256=sha(partial / 'track3_submission.parquet')),
        preparation=dict(script_sha256=sha(__file__), official_packer_passed=True, official_uploader_local_validation_passed=True, offline_preflight_passed=True),
        best_effort_episodes=[e for e in old.get('best_effort_episodes', []) if e not in swap], all_episodes_pass_local_reference_proxy=not [e for e in old.get('best_effort_episodes', []) if e not in swap],
        complete_official_roster=True, source_pose_proxy_not_official_ground_truth=True, hidden_ground_truth_scores_available=False, uploaded=False)
    write(partial / 'manifest.json', manifest)
    (partial / 'README.md').write_text((args.original_package / 'README.md').read_text() + f'''
## Replaced episodes ({stamp})

Episodes {', '.join(map(str, swap))} were replaced by freshly executed components under the same frozen
evaluator build. Reason: {args.reason}
The previous components' evidence is preserved under `evidence/replaced_{stamp}/`. All other
episodes' files are byte-identical to the previous package; the combined Parquet was re-packed by
the unchanged official packer and re-validated. Not uploaded.
''')
    inv = {str(p.relative_to(partial)): sha(p) for p in sorted(partial.rglob('*')) if p.is_file() and p != partial / 'SHA256SUMS.json'}
    write(partial / 'SHA256SUMS.json', inv); partial.replace(output)
    if args.archive:
        tmp = Path(str(outer) + '.partial')
        with tarfile.open(tmp, 'w') as tar: tar.add(output, arcname=output.name)
        tmp.replace(outer); write(Path(str(outer) + '.json'), dict(path=str(outer), bytes=outer.stat().st_size, sha256=sha(outer)))
    print(json.dumps(dict(output=str(output), submission=str(output / 'track3_submission.parquet'), replaced=swap, rows=len(frame), uploaded=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-package', type=Path, required=True); p.add_argument('--build-manifest', type=Path, default=base.DEFAULT_BASE / 'build_manifest.json')
    p.add_argument('--common-evaluator', type=Path, default=base.DEFAULT_BASE / 'package_code/common_evaluator.py'); p.add_argument('--kit', type=Path, default=base.DEFAULT_KIT)
    p.add_argument('--component', action='append', default=[]); p.add_argument('--portable', action='append', default=[]); p.add_argument('--video', action='append', default=[])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--reason', default='pot carry wrist correction'); p.add_argument('--archive', action='store_true')
    run(p.parse_args())
