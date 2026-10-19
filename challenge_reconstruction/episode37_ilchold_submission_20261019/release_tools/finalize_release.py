#!/usr/bin/env python3
"""Validate completed components, preserve episode37 evidence, assemble a new release.

This does not launch training/evaluation or upload anything. All thirteen common
executions must already have completed. Existing archives/releases are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

EPISODES = [1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38]
DEFAULT_BASE = Path('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019')
DEFAULT_OLD = Path('/data1/SRL_V2D/challenge_submissions/track3_dexagent_20261018/available_13_episodes')
DEFAULT_OUTPUT = Path('/data1/SRL_V2D/challenge_submissions/track3_mixed_ilchold_20261019/available_13_episodes')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_check(base):
    manifest = read(base / 'source_manifest.json')
    source = base / 'source/episode_000037'
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe source inventory path')
        path = source / rel
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Frozen source changed: ' + item['path'])
    video = source / 'demo/epi37.mp4'
    if sha(video) != manifest['selected_video_sha256']:
        raise ValueError('Frozen video is not the user-selected video')
    return source, manifest, video


def video_binding(component, source, video):
    """Compare original rendered body states with fresh recorded physical states."""
    import numpy as np

    completion = read(component / 'completed.json')
    report = completion['metrics']
    if (completion['backend_kind'] != 'frozen_uniretarget_ilchold' or
            not report['selected_world_strict_proxy_pass'] or
            report['selected_world_index'] != 28 or report['world_count'] != 32):
        raise ValueError('Episode37 is not the successful preselected user-video world')
    with np.load(completion['actions']['path'], allow_pickle=False) as z:
        fresh = z['body_q'].copy()
        for name, expected in [('time_s', np.arange(151) / 20.),
                               ('action_time_s', np.arange(150) / 20.),
                               ('timestep', np.arange(151)),
                               ('command_reference_frame', np.arange(150))]:
            if not np.allclose(z[name], expected, rtol=0, atol=1e-9):
                raise ValueError('Unexpected physical/command timeline: ' + name)
        if z['policy_action'].shape != (150, 56) or np.any(z['object_assistance_scale'] != 0):
            raise ValueError('Robot action/assistance contract failed')
        if np.any(z['terminated']) or np.any(z['truncated'][:-1]):
            raise ValueError('Premature termination in the selected recording')
    recording = source / 'demo/epi37_parts/rollouts.npz'
    with np.load(recording, allow_pickle=False) as z:
        original = z['refined'].copy()
        if int(z['world_refined']) != 28:
            raise ValueError('Original recording selected-world mismatch')
    if fresh.shape != original.shape or fresh.shape != (151, 54, 7):
        raise ValueError('Original/fresh body recording shapes differ')
    if not np.isfinite(fresh).all() or not np.isfinite(original).all():
        raise ValueError('Nonfinite video-binding evidence')
    dp = np.linalg.norm(fresh[..., :3] - original[..., :3], axis=-1)
    q0, q1 = fresh[..., 3:].astype(np.float64), original[..., 3:].astype(np.float64)
    q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 /= np.linalg.norm(q1, axis=-1, keepdims=True)
    degrees = np.degrees(2 * np.arccos(np.clip(np.abs((q0 * q1).sum(axis=-1)), 0, 1)))
    identical = bool(np.array_equal(fresh, original))
    max_pos, max_rot = float(dp.max()), float(degrees.max())
    passed = (max_pos <= 1e-5 and max_rot <= 1e-3 and
              report['original_demo_comparison'].get('numerically_matches_original_demo') is True)
    binding = dict(video_path=str(video), video_sha256=sha(video),
                   original_recording_sha256=sha(recording),
                   fresh_actions_sha256=sha(completion['actions']['path']),
                   fixed_world_index=28, body_states_bitwise_identical=identical,
                   max_body_position_difference_m=max_pos,
                   max_body_orientation_difference_deg=max_rot,
                   allowed_position_difference_m=1e-5, allowed_orientation_difference_deg=1e-3,
                   numerical_video_binding_passed=passed,
                   meaning='Fresh selected-world physical poses match the original rendered poses within explicit numerical tolerances; bitwise equality is reported separately.')
    if not passed:
        raise ValueError('Fresh trajectory differs from supplied video; render a fresh video before publishing. ' + json.dumps(binding))
    return binding


def portable_readme(image, commit):
    return f'''# Episode37 ilchold physical rollout

This bundle preserves the user-selected policy2508800/reference0005 run, its
scene and assets, original video, and a fresh unassisted action/state recording.
The original video world28 of32 was fixed before fresh evaluation. See
`video_binding.json` for numerical and bitwise agreement with the original video.

`component/backend/actions.npz` contains 151 states and 150 actual 56D actor actions
at 20 Hz. Physics uses 20 substeps (400 Hz). Actions are raw residual inputs, right
wrist 6/fingers 22 then left wrist 6/fingers 22, not absolute joint position targets.
State[k] and observation[k] produce policy_action[k], then state[k+1]. Processed
targets, joint states/velocities and action layout accompany the recording.
The last-substep generalized-force snapshot is not a full force time series.
All object assistance is disabled. Reference object poses are unchanged.

`component/completed.json` binds the checkpoint, native backend export and actual
actions to the common evaluator build. `common_build/` records source bytes and
the genuine code commit: {commit}
The twelve other package episodes keep their own legacy backend/action contract.
The MP-SR result uses reconstructed references, not official hidden ground truth.
Fresh execution and comparison to the original video were performed; this bundle
does not claim a second independent GPU execution of the extracted archive has
already been verified on another server.

## Reproduce a fresh GPU rollout

Extract this archive into its own directory. You need Docker with NVIDIA GPU
support and the exact image `{image}`. The image itself is not bundled. Copy it
from the source server with `docker save`/`docker load`, or obtain the identical
image through your registry. A local image SHA is not a downloadable registry URL.
Use the following commands from the extracted `episode_000037_portable` directory:

```bash
TASK_BUNDLE="$(pwd)"
TASK_OUTPUT="$(dirname "$TASK_BUNDLE")/episode37_replay"
mkdir -p "$TASK_OUTPUT"
docker run --rm --gpus all \\
  --user "$(id -u):$(id -g)" \\
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \\
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e WANDB_MODE=disabled \\
  -e OMP_NUM_THREADS=4 -e XDG_CACHE_HOME=/output/cache \\
  -e PYTHONPATH=/bundle/source/episode_000037/reproduction/packaging_python:/workspace/src \\
  -v "$TASK_BUNDLE:/bundle:ro" -v "$TASK_OUTPUT:/output" \\
  -v "$TASK_BUNDLE/source/episode_000037/flash_chord:/workspace:ro" \\
  -v "$TASK_BUNDLE/source/episode_000037/runtime_assets/meshes:/workspace/src/flash_chord/assets/meshes:ro" \\
  -v "$TASK_BUNDLE/source/episode_000037/runtime_assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro" \\
  -v "$TASK_BUNDLE/source/episode_000037/runtime_assets/policies:/workspace/src/flash_chord/assets/policies:ro" \\
  -w /workspace {image} \\
  python /bundle/code/export_ilchold_rollout.py \\
    --run-root /bundle/source/episode_000037 --output /output/fresh \\
    --runtime-root /workspace --world-count 32 --world-index 28
```

The output directory `/output/fresh` must not exist. The exporter
relocates asset path strings into its writable runtime-input copy and verifies
object target values stay unchanged. Read `fresh/report.json` for actual success,
native action semantics and original-video agreement. Hardware/runtime changes
can produce numerical differences; never substitute old actions for fresh poses.

`source_manifest.json` preserves the original source inventory and path strings;
`SHA256SUMS.json` hashes archive members by their extracted relative paths.
No source data is overwritten and no competition upload is performed.
'''


def build_portable(args, source, source_manifest, component, binding, build):
    archive = args.base / 'episode_000037_portable.tar.gz'
    if archive.exists():
        raise FileExistsError('Preserve existing portable archive: ' + str(archive))
    staging = args.base / 'portable_staging/episode_000037_portable'
    if staging.exists():
        raise FileExistsError('Preserve existing portable staging: ' + str(staging))
    staging.mkdir(parents=True)
    shutil.copytree(source, staging / 'source/episode_000037')
    shutil.copy2(args.base / 'source_manifest.json', staging / 'source_manifest.json')
    shutil.copytree(args.base / 'code', staging / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(component, staging / 'component', ignore=shutil.ignore_patterns('_cache', '__pycache__'))
    shutil.copytree(args.base / 'review', staging / 'review', ignore=shutil.ignore_patterns('__pycache__'))
    for record in build['sources']:
        destination = staging / 'common_build/sources' / record['key']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record['path'], destination)
    shutil.copy2(args.build_manifest, staging / 'common_build/build_manifest.json')
    write(staging / 'video_binding.json', binding)
    (staging / 'README.md').write_text(portable_readme(source_manifest['docker_image'], build['code_commit_url']))
    for item in source_manifest['files']:
        if sha(staging / 'source/episode_000037' / item['path']) != item['sha256']:
            raise ValueError('Portable copy changed source bytes: ' + item['path'])
    inventory = {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    write(staging / 'SHA256SUMS.json', inventory)
    temporary = archive.with_name(archive.name + '.partial')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as packed:
        packed.add(staging, arcname=staging.name)
    temporary.replace(archive)
    write(Path(str(archive) + '.json'), dict(path=str(archive), bytes=archive.stat().st_size,
                                           sha256=sha(archive), member_hashes=len(inventory)))
    return archive


def run(args):
    sys.path.insert(0, str(args.base / 'package_code'))
    from common_evaluator import verify_build, official_validate, validate_meshes

    if args.output.exists() or Path(str(args.output) + '.tar').exists():
        raise FileExistsError('Preserve existing release: ' + str(args.output))
    missing = [f'episode_{e:06d}' for e in EPISODES
               if not (args.components_root / f'episode_{e:06d}/completed.json').is_file()]
    if missing:
        raise ValueError('Fresh common components incomplete: ' + ', '.join(missing))
    build = verify_build(args.build_manifest)
    for episode in EPISODES:
        directory = args.components_root / f'episode_{episode:06d}'
        report = read(directory / 'completed.json')
        if (not report['fresh_backend_executed'] or
                report['build_manifest_sha256'] != sha(args.build_manifest) or
                report['eval_code_sha256'] != build['eval_code_sha256'] or
                report['eval_script_sha256'] != build['eval_script_sha256'] or
                report['code_commit_url'] != build['code_commit_url']):
            raise ValueError('Common execution/build mismatch: ' + str(episode))
        if episode != 37 and not report.get('exact_previous_physical_states_actions'):
            raise ValueError('Legacy physical replay mismatch: ' + str(episode))
        for key in ('checkpoint', 'actions', 'backend_export', 'trajectory'):
            if sha(report[key]['path']) != report[key]['sha256']:
                raise ValueError('Changed completed artifact: ' + str(episode) + '/' + key)
        validation = official_validate(Path(report['trajectory']['path']), args.kit,
                                       episode, report['checkpoint']['sha256'])
        validate_meshes(directory / 'meshes', args.kit, episode, validation['object_names'])
    source, source_manifest, video = source_check(args.base)
    component = args.components_root / 'episode_000037'
    binding = video_binding(component, source, video)
    write(args.base / 'video_binding.json', binding)
    archive = build_portable(args, source, source_manifest, component, binding, build)
    command = [sys.executable, str(args.base / 'package_code/assemble_mixed_package.py'),
               '--build-manifest', str(args.build_manifest), '--original-package', str(args.original_package),
               '--components-root', str(args.components_root), '--kit', str(args.kit), '--output', str(args.output),
               '--episode37-archive', str(archive), '--episode37-video', str(video),
               '--episode37-evidence', str(args.base / 'review'), '--archive']
    subprocess.run(command, check=True)
    manifest = read(args.output / 'manifest.json')
    if manifest['available_episodes'] != EPISODES or not manifest['available_episode_components_ready']:
        raise ValueError('Assembled release scope/readiness mismatch')
    inventory = read(args.output / 'SHA256SUMS.json')
    for relative, expected in inventory.items():
        if sha(args.output / relative) != expected:
            raise ValueError('Assembled member hash mismatch: ' + relative)
    transport = Path(str(args.output) + '.tar')
    transport_meta = read(Path(str(transport) + '.json'))
    if sha(transport) != transport_meta['sha256']:
        raise ValueError('Transport archive hash mismatch')
    status = dict(stage='replacement_package_completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  episode='episode_000037', selected_policy='policy_2508800.safetensors', selected_reference='reference_0005.npz',
                  source_snapshot_verified=True, fresh_gpu_replay_completed=True,
                  fresh_common_components_completed=EPISODES, replacement_package_completed=True,
                  package=str(args.output), transport_archive=transport_meta, portable_archive=str(archive),
                  video_binding=binding, original_13_package_preserved=True,
                  official_complete_twenty_episode_submission_ready=False, uploaded=False)
    write(args.status, status)
    print(json.dumps(status, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=DEFAULT_BASE)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--build-manifest', type=Path)
    p.add_argument('--components-root', type=Path)
    p.add_argument('--original-package', type=Path, default=DEFAULT_OLD)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--status', type=Path)
    args = p.parse_args()
    args.base = args.base.resolve()
    args.build_manifest = args.build_manifest or args.base / 'build_manifest.json'
    args.components_root = args.components_root or args.base / 'common_components'
    args.status = args.status or args.base / 'release_status.json'
    try:
        run(args)
    except Exception as exc:
        write(args.status, dict(stage='replacement_finalization_failed', error=f'{type(exc).__name__}: {exc}',
                                replacement_package_completed=False, original_13_package_preserved=True,
                                output_requested=str(args.output), output_exists=args.output.exists(), uploaded=False))
        raise


if __name__ == '__main__':
    main()
