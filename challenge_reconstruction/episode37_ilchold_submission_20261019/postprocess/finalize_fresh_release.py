#!/usr/bin/env python3
"""Package a successful fresh rollout with its own correctly bound video.

The selected historical video and fresh replay differ numerically. Both survive,
with distinct labels. No evaluation source, physics recording, manifest or
producing evaluator hash is changed by this postprocessing step.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

DEFAULT_BASE = Path('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019')
DEFAULT_OLD = Path('/data1/SRL_V2D/challenge_submissions/track3_dexagent_20261018/available_13_episodes')
DEFAULT_OUTPUT = Path('/data1/SRL_V2D/challenge_submissions/track3_mixed_ilchold_20261019/available_13_episodes')
EPISODES = [1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_frozen_finalizer(base):
    path = base / 'release_tools/finalize_release.py'
    spec = importlib.util.spec_from_file_location('frozen_release_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_video_file(path):
    """Check the complete video even when system ffprobe/ffmpeg are absent."""
    probe_binary = shutil.which('ffprobe')
    if probe_binary:
        probe = json.loads(subprocess.check_output([probe_binary, '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames,avg_frame_rate,duration',
            '-of', 'json', str(path)], text=True))['streams'][0]
        numerator, denominator = map(int, probe['avg_frame_rate'].split('/'))
        frames, fps = int(probe['nb_read_frames']), numerator / denominator
        reader = 'ffprobe -count_frames'
    else:
        import cv2
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError('Cannot open fresh video for complete decoding')
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise ValueError('Empty decoded frame in fresh video')
                frames += 1
        finally:
            capture.release()
        reader = 'OpenCV full frame decode (no ffprobe installed)'
    if frames != 151 or abs(fps - 20.) > 1e-9:
        raise ValueError(f'Rendered video does not preserve 151 frames at 20 Hz: {frames=}, {fps=}')
    decoder = shutil.which('ffmpeg')
    if not decoder:
        import imageio_ffmpeg
        decoder = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([decoder, '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                   check=True, stdout=subprocess.DEVNULL)
    return dict(decoded_frames=frames, fps=fps, frame_reader=reader,
                complete_ffmpeg_decode_passed=True, ffmpeg_binary=str(decoder))


def validate_video(args, component, original_video):
    import numpy as np
    report = read(component / 'completed.json')
    metrics = report['metrics']
    if (report['backend_kind'] != 'frozen_uniretarget_ilchold' or
            not metrics['selected_world_strict_proxy_pass'] or
            metrics['selected_world_index'] != 28 or metrics['world_count'] != 32):
        raise ValueError('Expected the successful preselected video world 28 of 32')
    evidence = read(args.video_evidence)
    if (evidence.get('video_sha256') != sha(args.fresh_video) or
            evidence.get('actions_sha256') != report['actions']['sha256'] or
            evidence.get('rendered_from_fresh_recording') is not True or
            evidence.get('frame_count') != 151 or evidence.get('fps') != 20. or
            evidence.get('original_video_sha256') != sha(original_video)):
        raise ValueError('Fresh video evidence does not identify the actual recording/video')
    with np.load(report['actions']['path'], allow_pickle=False) as data:
        if data['body_q'].shape != (151, 54, 7) or data['policy_action'].shape != (150, 56):
            raise ValueError('Physical state/action shape mismatch')
        if not np.isfinite(data['body_q']).all() or not np.isfinite(data['policy_action']).all():
            raise ValueError('Non-finite physical state/action recording')
        for name, expected in [('time_s', np.arange(151) / 20.),
                               ('action_time_s', np.arange(150) / 20.),
                               ('timestep', np.arange(151)),
                               ('command_reference_frame', np.arange(150))]:
            if not np.allclose(data[name], expected, rtol=0, atol=1e-9):
                raise ValueError('Physical/command timeline mismatch: ' + name)
        if np.any(data['object_assistance_scale']) or np.any(data['terminated']) or np.any(data['truncated'][:-1]):
            raise ValueError('Object assistance or premature termination is present')
        body_sha = hashlib.sha256(np.ascontiguousarray(data['body_q']).tobytes()).hexdigest()
        if evidence.get('body_q_sha256') != body_sha:
            raise ValueError('Video body poses are not the fresh physical recording')
    video_validation = validate_video_file(args.fresh_video)
    return dict(schema='source_and_fresh_video_binding_v1',
        fresh_video_path=str(args.fresh_video.resolve()), fresh_video_sha256=sha(args.fresh_video),
        fresh_render_evidence_sha256=sha(args.video_evidence), actions_sha256=report['actions']['sha256'],
        body_q_sha256=body_sha, frame_count=151, fps=20.,
        original_video_path=str(original_video), original_video_sha256=sha(original_video),
        original_video_is_historical_evidence=True, fresh_video_matches_exported_recording=True,
        fresh_recording_matches_original_video=metrics['original_demo_comparison']['numerically_matches_original_demo'],
        video_validation=video_validation,
        original_demo_comparison=metrics['original_demo_comparison'],
        explanation='The same pinned checkpoint/reference pair was freshly evaluated. Its numerical execution differs from the historical video; the package uses a separately rendered video of the new actual recording.')


def archive_directory(directory, archive):
    if archive.exists() or Path(str(archive) + '.partial').exists():
        raise FileExistsError('Preserve existing archive: ' + str(archive))
    temporary = Path(str(archive) + '.partial')
    mode = 'w:gz' if str(archive).endswith('.gz') else 'w'
    kwargs = {'compresslevel': 3} if mode == 'w:gz' else {}
    with tarfile.open(temporary, mode, **kwargs) as packed:
        packed.add(directory, arcname=directory.name)
    temporary.replace(archive)
    metadata = dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))
    write(Path(str(archive) + '.json'), metadata)
    return metadata


def refresh_inventory(directory):
    inventory = {str(path.relative_to(directory)): sha(path) for path in sorted(directory.rglob('*'))
                 if path.is_file() and path != directory / 'SHA256SUMS.json'}
    write(directory / 'SHA256SUMS.json', inventory)
    return inventory


def build_portable(args, frozen, source, source_manifest, component, binding, build, postprocess):
    staging = args.base / 'portable_staging_v2/episode_000037_portable'
    archive = args.base / 'episode_000037_portable_v2.tar.gz'
    if staging.exists() or archive.exists():
        raise FileExistsError('Preserve existing fresh-video portable artifacts')
    staging.mkdir(parents=True)
    shutil.copytree(source, staging / 'source/episode_000037')
    shutil.copy2(args.base / 'source_manifest.json', staging / 'source_manifest.json')
    shutil.copytree(args.base / 'code', staging / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(component, staging / 'component', ignore=shutil.ignore_patterns('_cache', '__pycache__'))
    shutil.copytree(args.base / 'review', staging / 'review', ignore=shutil.ignore_patterns('__pycache__'))
    for entry in build['sources']:
        destination = staging / 'common_build/sources' / entry['key']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry['path'], destination)
    shutil.copy2(args.build_manifest, staging / 'common_build/build_manifest.json')
    videos = staging / 'videos'
    videos.mkdir()
    shutil.copy2(args.fresh_video, videos / 'fresh_physics_replay.mp4')
    shutil.copy2(source / 'demo/epi37.mp4', videos / 'historical_user_video.mp4')
    shutil.copy2(args.video_evidence, videos / 'fresh_video_evidence.json')
    write(staging / 'video_binding.json', binding)
    write(staging / 'postprocess_provenance.json', postprocess)
    shutil.copytree(args.base / 'postprocess', staging / 'postprocess', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(args.base / 'render_tools', staging / 'render_tools', ignore=shutil.ignore_patterns('__pycache__'))
    original_readme = frozen.portable_readme(source_manifest['docker_image'], build['code_commit_url'])
    reproduction = original_readme.split('## Reproduce a fresh GPU rollout', 1)[1]
    reproduction = reproduction.replace(
        'mkdir -p "$TASK_OUTPUT"\ndocker run --rm --gpus all',
        'TASK_GPU=0  # Replace with your allocated GPU index or UUID.\n'
        'mkdir -p "$TASK_OUTPUT"\n'
        'mkdir -p "$TASK_BUNDLE/source/episode_000037/flash_chord/src/flash_chord/assets/"{meshes,urdfs,policies}\n'
        'docker run --rm --gpus "device=$TASK_GPU"')
    reproduction += '''
The mountpoint directories above are empty placeholders in the extracted source
copy. Docker overlays the bundled `runtime_assets/` on them while keeping source
code read-only. They are not replacement meshes. The supplied `PYTHONPATH`
exposes the bundled packaging dependencies and the pinned FlashCHORD source;
do not install another NumPy/JAX version into the image.

Convex-decomposition cache entries are copied from the bundle's `asset_cache/`
into the new output. Warp kernels compile into `/output/fresh/_cache/warp`;
the original server's Warp binaries are not required. The first execution can
therefore spend extra time compiling. `/output` must be writable by the selected
container UID. No `/data1` or `/data3` host mount is needed for policy inference:
legacy source-path strings remain historical metadata, while inference uses
bundled pose arrays, relocated assets, and the included finger-basis files.
'''
    (staging / 'README.md').write_text('''# Episode 37: ilchold policy and fresh physical replay

The user-selected policy_2508800.safetensors and reference_0005.npz are preserved.
World 28 of 32 was fixed from the supplied video's metadata before evaluation.
The fresh GPU rollout passes full-frame reference-proxy MP-SR, but its body states
are not numerically identical to the historical recording. This package does not
represent the historical video as a rendering of the fresh submission trajectory.

- `videos/fresh_physics_replay.mp4` shows the actual fresh exported recording.
- `videos/historical_user_video.mp4` preserves the supplied epi37.mp4 as historical evidence.
- `video_binding.json` records the measured original/fresh differences.
- `videos/fresh_video_evidence.json` binds the new video to the actual actions and body poses.

The fresh video's physics panel corresponds to `component/backend/actions.npz`.
Any historical baseline panel is labeled separately. That recording contains
151 states and 150 causal, 56-dimensional actor residual actions at 20 Hz;
simulation uses 400 Hz physics. These are residual policy actions, not the
500 Hz absolute position targets used by the other twelve legacy controllers.
Processed targets and joint/body states accompany the raw actions. Object
assistance is zero, and original object target trajectories remain unchanged.

The producing evaluator build and its pinned commit are unchanged. Rendering
and packaging provenance is recorded separately in `postprocess_provenance.json`.
Proxy scores use reconstructed references, not confidential official ground truth.
Fresh execution has been verified on the source server; a second independent
execution of this extracted archive on another server is not claimed.

## Reproduce a fresh GPU rollout''' + reproduction)
    for item in source_manifest['files']:
        if sha(staging / 'source/episode_000037' / item['path']) != item['sha256']:
            raise ValueError('Portable copy changed source bytes: ' + item['path'])
    for path, expected in [(staging / 'component/backend/actions.npz', binding['actions_sha256']),
                           (videos / 'fresh_physics_replay.mp4', binding['fresh_video_sha256']),
                           (videos / 'historical_user_video.mp4', binding['original_video_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Portable copy changed bound actions/video')
    refresh_inventory(staging)
    archive_directory(staging, archive)
    return archive


def run(args):
    sys.path.insert(0, str(args.base / 'package_code'))
    from common_evaluator import verify_build, official_validate, validate_meshes
    build = verify_build(args.build_manifest)
    frozen = load_frozen_finalizer(args.base)
    if args.output.exists() or Path(str(args.output) + '.tar').exists():
        raise FileExistsError('Preserve existing release: ' + str(args.output))
    for episode in EPISODES:
        directory = args.components_root / f'episode_{episode:06d}'
        report = read(directory / 'completed.json')
        if (not report['fresh_backend_executed'] or report['build_manifest_sha256'] != sha(args.build_manifest) or
                report['eval_code_sha256'] != build['eval_code_sha256'] or
                report['eval_script_sha256'] != build['eval_script_sha256'] or
                report['code_commit_url'] != build['code_commit_url']):
            raise ValueError('Completed component/build mismatch: ' + str(episode))
        if episode != 37 and not report.get('exact_previous_physical_states_actions'):
            raise ValueError('Legacy physical replay mismatch: ' + str(episode))
        for key in ('checkpoint', 'actions', 'backend_export', 'trajectory'):
            if sha(report[key]['path']) != report[key]['sha256']:
                raise ValueError('Changed completed component artifact: ' + str(episode) + '/' + key)
        validation = official_validate(Path(report['trajectory']['path']), args.kit, episode, report['checkpoint']['sha256'])
        validate_meshes(directory / 'meshes', args.kit, episode, validation['object_names'])
    source, source_manifest, original_video = frozen.source_check(args.base)
    component = args.components_root / 'episode_000037'
    binding = validate_video(args, component, original_video)
    postprocess = dict(scope='Rendering and archive assembly only; no policy/physical-state/evaluator modifications',
        evaluator_code_commit_url=build['code_commit_url'], postprocess_code_commit_url=args.postprocess_commit_url,
        finalizer_script_sha256=sha(__file__), renderer_script_sha256=sha(args.base / 'render_tools/render_fresh.py'),
        render_evidence_sha256=sha(args.video_evidence), evaluator_build_manifest_sha256=sha(args.build_manifest))
    archive = build_portable(args, frozen, source, source_manifest, component, binding, build, postprocess)
    command = [sys.executable, str(args.base / 'package_code/assemble_mixed_package.py'),
        '--build-manifest', str(args.build_manifest), '--original-package', str(args.original_package),
        '--components-root', str(args.components_root), '--kit', str(args.kit), '--output', str(args.output),
        '--episode37-archive', str(archive), '--episode37-video', str(args.fresh_video),
        '--episode37-video-evidence', str(args.video_evidence), '--episode37-evidence', str(args.base / 'review')]
    subprocess.run(command, check=True)
    # This new release has not yet been archived or published. Add clearly labeled
    # historical evidence, then seal its complete final inventory exactly once.
    video_directory = args.output / 'videos/episode_000037'
    shutil.copy2(original_video, video_directory / 'historical_user_video.mp4')
    (video_directory / 'README.md').write_text('''`epi37.mp4` is the freshly rendered physical recording bound to this package's
robot_actions/episode_000037.npz. Its comparison panels are labeled individually.
`historical_user_video.mp4` is the user's original supplied video. The same
checkpoint/reference pair generated both executions, but their numerical body
states differ. See evidence/episode_000037/source_and_fresh_video_binding.json.
''')
    evidence = args.output / 'evidence/episode_000037'
    write(evidence / 'source_and_fresh_video_binding.json', binding)
    write(evidence / 'postprocess_provenance.json', postprocess)
    manifest = read(args.output / 'manifest.json')
    if manifest['available_episodes'] != EPISODES or not manifest['available_episode_components_ready']:
        raise ValueError('Assembled scope/readiness mismatch')
    manifest['episode37_video_binding'] = binding
    manifest['postprocess_provenance'] = postprocess
    write(args.output / 'manifest.json', manifest)
    with (args.output / 'README.md').open('a') as stream:
        stream.write('\nEpisode 37 includes a new video rendered from its actual exported physical\n'
                     'recording. The original supplied video is retained separately as historical\n'
                     'evidence because the two executions differ numerically. See its video README\n'
                     'and evidence for the exact binding. Evaluation code and physics are unchanged.\n')
    refresh_inventory(args.output)
    verify_build(args.build_manifest)
    transport = archive_directory(args.output, Path(str(args.output) + '.tar'))
    status = dict(stage='replacement_package_completed', completed_utc=datetime.now(timezone.utc).isoformat(),
        replacement_package_completed=True, fresh_common_components_completed=EPISODES,
        source_snapshot_verified=True, fresh_gpu_replay_completed=True,
        package=str(args.output), transport_archive=transport, portable_archive=str(archive),
        episode37_source_video_preserved=True, episode37_fresh_matching_video_included=True,
        video_binding=binding, evaluator_commit_unchanged=build['code_commit_url'],
        original_13_package_preserved=True, official_complete_twenty_episode_submission_ready=False, uploaded=False)
    write(args.status, status)
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path)
    parser.add_argument('--components-root', type=Path)
    parser.add_argument('--original-package', type=Path, default=DEFAULT_OLD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fresh-video', type=Path)
    parser.add_argument('--video-evidence', type=Path)
    parser.add_argument('--postprocess-commit-url')
    parser.add_argument('--status', type=Path)
    args = parser.parse_args()
    args.base = args.base.resolve()
    args.build_manifest = args.build_manifest or args.base / 'build_manifest.json'
    args.components_root = args.components_root or args.base / 'common_components'
    args.fresh_video = args.fresh_video or args.base / 'rendered_fresh/epi37_fresh.mp4'
    args.video_evidence = args.video_evidence or args.base / 'rendered_fresh/epi37_fresh.video_evidence.json'
    args.status = args.status or args.base / 'release_status_fresh_video.json'
    if args.postprocess_commit_url and not re.fullmatch(r'https://[\w.-]*github[\w.-]*/[\w.-]+/[\w.-]+/commit/[0-9a-fA-F]{7,40}/?', args.postprocess_commit_url):
        parser.error('--postprocess-commit-url must identify an actual GitHub commit')
    try:
        run(args)
    except Exception as error:
        write(args.status, dict(stage='fresh_video_finalization_failed', error=f'{type(error).__name__}: {error}',
            replacement_package_completed=False, original_13_package_preserved=True,
            output_requested=str(args.output), output_exists=args.output.exists(), uploaded=False))
        raise


if __name__ == '__main__':
    main()
