"""Collect all available source episodes without altering the official roster.

Requires completed common-build components. Produces a portable working package,
not a leaderboard submission with fabricated missing episodes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import tarfile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_component(name, common, source, kit, roster):
    """Bind the current exported bytes to the completed, reviewed common build."""
    from submission_readiness import trajectory_info

    result = read(common/name/'completed.json')
    component = Path(result['component']).resolve()
    if component != (common/name/'component').resolve() or result['episode'] != name:
        raise ValueError('Common component points to another episode: '+name)
    ready = read(component/'readiness.json')
    replay = read(component/'portable_validation/portable_validation.json')
    producing = read(component/'provenance/producing_code.json')
    bundle = read(component/'portable/bundle_manifest.json')
    if not (ready['episode'] == replay['episode'] == bundle['episode'] == name and
            ready['episode_ready_for_packaging_review'] and
            result['original_reviewed_execution_exact'] and result['portable_execution_exact'] and
            replay['all_physical_states_actions_exact'] and replay['source_object_targets_unchanged'] and
            replay['compiled_model_arrays_options_exact'] and replay['original_weights_unchanged'] and
            replay['independent_control_replay']['ok'] and
            ready['reference_proxy_mp_sr'] == replay['reference_proxy_mp_sr'] == result['reference_proxy_mp_sr'] == 1):
        raise ValueError('Incomplete episode verification: '+name)
    if sha(component/'readiness.json') != result['readiness_sha256']:
        raise ValueError('Common component readiness changed: '+name)
    if (Path(ready['official_kit']).resolve() != kit or
            sha(kit/'KIT_MANIFEST.json') != ready['official_kit_manifest_sha256']):
        raise ValueError('Public kit differs between components: '+name)
    checkpoint = ready['learned_policy_checkpoint']
    if not (sha(checkpoint['path']) == checkpoint['sha256'] ==
            sha(component/'portable/original/policy.npz') == result['checkpoint_sha256'] ==
            producing['checkpoint_sha256'] == replay['original_checkpoint_sha256'] ==
            bundle['original_checkpoint_sha256']):
        raise ValueError('Checkpoint provenance mismatch: '+name)
    if producing['commit'] != result['producing_commit'] or not re.fullmatch(r'[0-9a-f]{40}', producing['commit']):
        raise ValueError('Common producing commit mismatch: '+name)
    code_digest = hashlib.sha256()
    for entry in ready['evaluation_code_manifest']:
        path = Path(entry['path']).resolve()
        verified = [record for record in producing['actual_sources_verified']
                    if path.as_posix().endswith('/'+record['path'])]
        if (len(verified) != 1 or not verified[0]['git_bytes_match'] or
                verified[0]['sha256'] != entry['sha256'] or sha(path) != entry['sha256']):
            raise ValueError('Producing evaluator bytes changed: '+str(path))
        code_digest.update(path.read_bytes())
    if code_digest.hexdigest() != ready['evaluation_code_sha256'] or result['eval_code_sha256'] != ready['evaluation_code_sha256']:
        raise ValueError('Common evaluation digest mismatch: '+name)
    for target, key in [('object_poses.npz', 'source_object_poses_sha256'),
                        ('reference.parquet', 'source_reference_sha256')]:
        if not (sha(source/name/target) == ready[key] == bundle[key]):
            raise ValueError('Source target changed: '+name+'/'+target)
    trajectory = ready['policy_rollout']
    trajectory_path = component/'results'/(name+'.parquet')
    if Path(trajectory['path']).resolve() != trajectory_path or sha(trajectory_path) != trajectory['sha256']:
        raise ValueError('Achieved trajectory changed: '+name)
    expected = dict(roster[int(name.split('_')[-1])], fps=ready['fps'])
    actual = trajectory_info(trajectory_path, expected, kit, checkpoint)
    if (actual['problems'] or actual['rows'] != ready['expected_trajectory_rows'] or
            actual['provenance']['eval_code_sha256'] != ready['evaluation_code_sha256'] or
            actual['provenance']['eval_script_sha256'] != ready['evaluator_script_sha256']):
        raise ValueError('Current public schema/provenance check failed: '+name+' '+str(actual))
    if set(ready['meshes']) != set(expected['object_names']):
        raise ValueError('Exported mesh roster mismatch: '+name)
    expected_meshes = set()
    for mesh in ready['meshes'].values():
        path = Path(mesh['staged']).resolve()
        if path.parent != component/'meshes'/name or sha(path) != mesh['sha256']:
            raise ValueError('Exported geometry changed: '+str(path))
        expected_meshes.add(path)
    for directory, expected_files in [('results', {trajectory_path}), ('meshes', expected_meshes)]:
        if {p.resolve() for p in (component/directory).rglob('*') if p.is_file()} != expected_files:
            raise ValueError('Unexpected exported files in '+name+'/'+directory)
    if read(component/'archive.json') != result['archive']:
        raise ValueError('Archive record changed: '+name)
    archive = Path(result['archive']['path']).resolve()
    if (archive.parent != component or archive.name != name+'_portable.tar.gz' or
            archive.stat().st_size != result['archive']['bytes'] or sha(archive) != result['archive']['sha256']):
        raise ValueError('Portable archive changed: '+name)
    # Recheck the evidence copied into the aggregate against the immutable archive.
    # This catches stale/tampered audits or meshes independently of their filenames.
    expected_archived = {component/'readiness.json', component/'README.md',
                         component/'portable_validation/portable_validation.json'}
    for directory in ['results', 'meshes', 'validation', 'provenance']:
        expected_archived.update(p for p in (component/directory).rglob('*') if p.is_file())
    archive_hashes = {str(Path(component.name)/p.relative_to(component)): sha(p) for p in expected_archived}
    found = set()
    with tarfile.open(archive, 'r|gz') as packed:
        for member in packed:
            if member.name not in archive_hashes:
                continue
            if not member.isfile() or member.name in found:
                raise ValueError('Invalid/duplicate archived evidence: '+member.name)
            digest = hashlib.sha256()
            stream = packed.extractfile(member)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != archive_hashes[member.name]:
                raise ValueError('Current evidence differs from archive: '+member.name)
            found.add(member.name)
    if found != set(archive_hashes):
        raise ValueError('Evidence missing from portable archive: '+str(sorted(set(archive_hashes)-found)))
    return name, component, ready, result


def run(common_root, source_root, output=None, archive=False, validate_only=False):
    common, source = map(lambda x: Path(x).resolve(), [common_root, source_root])
    directories = sorted(p for p in source.glob('episode_*') if p.is_dir())
    for directory in directories:
        if not re.fullmatch(r'episode_\d{6}', directory.name) or not (directory/'manifest.json').is_file():
            raise ValueError('Do not silently omit an incomplete source episode: '+str(directory))
    episodes = [p.name for p in directories]
    if not episodes:
        raise ValueError('Source folder has no episode manifests')
    output = Path(output).resolve() if output else None
    if not validate_only and output is None:
        raise ValueError('--output is required for assembly')
    if not validate_only and output.exists():
        raise FileExistsError('Preserve existing package: '+str(output))
    missing_components = [name for name in episodes if not (common/name/'completed.json').is_file()]
    if missing_components and not validate_only:
        raise ValueError('Requested source episodes lack completed components: '+', '.join(missing_components))
    completed = [name for name in episodes if name not in missing_components]
    if not completed:
        raise ValueError('No completed common-build component to validate')
    first = read(common/completed[0]/'completed.json')
    kit = Path(read(Path(first['component'])/'readiness.json')['official_kit']).resolve()
    from submission_readiness import public_roster, kit_info
    if kit_info(kit)['manifest_file_mismatches']:
        raise ValueError('Frozen public submission kit contents changed')
    roster = public_roster(kit)
    available_ids = [int(name.split('_')[-1]) for name in episodes]
    if set(available_ids)-set(roster):
        raise ValueError('Source includes episodes outside the unchanged official roster')
    # Validate the complete requested scope before copying any package files.
    selected = []
    for name in completed:
        selected.append(validate_component(name, common, source, kit, roster))
    fields = {
        'eval_code_sha256': {r['evaluation_code_sha256'] for _, _, r, _ in selected},
        'eval_script_sha256': {r['evaluator_script_sha256'] for _, _, r, _ in selected},
        'common_evaluation_commit': {d['producing_commit'] for _, _, _, d in selected},
    }
    if any(len(values) != 1 for values in fields.values()):
        raise ValueError('Episodes were not built with one common evaluator: '+str(fields))
    missing = sorted(set(roster)-set(available_ids))
    if validate_only:
        print(json.dumps(dict(validated_components=completed, missing_completed_components=missing_components,
            requested_source_scope_complete=not missing_components, official_required_episodes=sorted(roster),
            missing_official_source_episodes=missing, no_package_written=True, uploaded=False)))
        return
    if archive and Path(str(output)+'.tar').exists():
        raise FileExistsError(Path(str(output)+'.tar'))
    output.mkdir(parents=True)
    rows = []
    for name, component, ready, result in selected:
        for directory in ['results', 'meshes', 'videos']:
            if (component/directory).exists():
                destination = output/directory if directory != 'videos' else output/'videos'/name
                shutil.copytree(component/directory, destination, dirs_exist_ok=True)
        archive_path = output/'portable_archives'/Path(result['archive']['path']).name
        archive_path.parent.mkdir(exist_ok=True)
        shutil.copy2(result['archive']['path'], archive_path)
        review = output/'evidence'/name
        review.mkdir(parents=True)
        for file in ['readiness.json', 'README.md', 'archive.json']:
            shutil.copy2(component/file, review/file)
        for directory in ['validation', 'provenance']:
            shutil.copytree(component/directory, review/directory)
        metrics = read(component/'validation/reference_proxy_mp_sr.json')['official_style_reference_proxy']['objects']
        rows.append(dict(episode=name, expected_rows=ready['expected_trajectory_rows'],
            checkpoint_sha256=ready['learned_policy_checkpoint']['sha256'],
            reference_proxy_mp_sr=1, objects=metrics,
            portable_archive=str(archive_path.relative_to(output)), archive_sha256=sha(archive_path),
            common_component=str(component), original_targets_unchanged=True))
    manifest = dict(scope='All episodes present in the requested final_recon folder',
        available_episodes=available_ids, requested_episode_count=len(episodes),
        available_episode_components_ready=True, official_required_episode_count=len(roster),
        official_required_episodes=sorted(roster), official_kit_manifest_sha256=sha(kit/'KIT_MANIFEST.json'),
        missing_official_source_episodes=missing, official_roster_unchanged=True,
        complete_official_submission_ready=False, uploaded=False,
        source_pose_proxy_not_official_ground_truth=True,
        evaluator={key: next(iter(values)) for key, values in fields.items()}, episodes=rows)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# Available Track 3 episode components', '',
        f'All {len(episodes)} episodes present in the requested source folder have verified learned-policy components.',
        'Each passes reference-proxy MP-SR and the public episode format. The original',
        'targets are unchanged, and every selected rollout reproduces exactly under',
        'the common evaluator and its portable execution. These are custom learned',
        'MuJoCo controllers, not organizer-certified FlashCHORD execution.', '',
        'The original source reconstructions are proxy ground truth; these are not',
        'confidential leaderboard scores. Reviewed finger, slip, release and alternate',
        'hand-role limitations are retained under `evidence/` for every episode.', '',
        '| Episode | Object | Mean position | Mean orientation |',
        '| --- | --- | ---: | ---: |']
    for row in rows:
        for obj, metric in row['objects'].items():
            lines.append(f"| {row['episode']} | {obj} | {metric['mean_position_error_cm'][0]:.3f} cm | {metric['mean_orientation_error_degrees'][0]:.3f} deg |")
    lines += ['', '`results/` contains achieved-pose Parquet files; `meshes/` contains the',
        'exact versioned visual meshes used for export. `portable_archives/` contains',
        'one self-contained actor/reproduction archive per episode. `videos/` contains',
        'the reviewed comparisons. No reference pose is substituted for achieved physics.', '',
        'Extract each episode archive into its own directory. For example:', '',
        '```sh',
        'mkdir -p /tmp/episode8',
        'tar -xzf portable_archives/episode_000008_portable.tar.gz -C /tmp/episode8',
        'python3.11 -m venv /tmp/episode8/venv',
        '/tmp/episode8/venv/bin/pip install -r /tmp/episode8/component/portable/requirements.txt',
        '/tmp/episode8/venv/bin/python /tmp/episode8/component/portable/code/reproduce_policy_component.py replay --bundle /tmp/episode8/component/portable --output /tmp/episode8/replay',
        '```', '',
        'Each archive uses a `component/` root. Keep episodes in separate extraction',
        'directories, and inspect the resulting `portable_validation.json` on new hardware.', '',
        'The unmodified official roster also requires episodes '+', '.join(map(str, missing))+'.',
        'They are absent from the requested source folder. This package therefore is',
        'not a complete leaderboard submission; do not upload it as one or fill missing',
        'rows with references. The official packer and full validator must be run after',
        'the remaining source episodes have valid components. Nothing has been uploaded.', '']
    (output/'README.md').write_text('\n'.join(lines))
    inventory = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'SHA256SUMS.json').write_text(json.dumps(inventory, indent=2)+'\n')
    if archive:
        path = Path(str(output)+'.tar')
        if path.exists():
            raise FileExistsError(path)
        # Episode archives/videos are already compressed; avoid expensive recompression.
        with tarfile.open(path, 'w') as packed:
            packed.add(output, arcname=output.name)
        record = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        path.with_suffix('.tar.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), available_components=len(rows),
                         missing_official_source_episodes=missing, uploaded=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['common-root', 'source-root']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--output')
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Check existing completed components; report missing episodes without creating a partial package')
    run(**vars(parser.parse_args()))
