#!/usr/bin/env python3
"""Render the fresh selected-world recording using the frozen demo renderer.

No policy inference or physics rollout is performed. Source simulation body
states are copied without edits into the renderer's recorded-state input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019')
EGO = Path('/data1/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000037.mp4')
CAMERA = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000037/camera_motion.npz')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def count_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count, shape = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        shape = list(frame.shape)
    cap.release()
    if count != 151 or abs(fps - 20.) > 1e-6:
        raise ValueError(f'Incomplete video {path}: frames={count}, fps={fps}')
    return dict(frames=count, fps=fps, decoded_final_frame_shape=shape)


def label_panels(source, destination, parts):
    """Replace only the existing header, making old baseline/fresh rollout explicit."""
    import cv2
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (2560, 542):
        raise ValueError(f'Unexpected frozen four-panel layout: {(width, height)}')
    raw = parts / 'labeled_raw.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 20., (width, height))
    if not writer.isOpened():
        raise RuntimeError('Could not open header-label video encoder')
    titles = [('Original human video', 'source ego camera'),
              ('Reference (kinematic)', 'reference0005; blue = original hands'),
              ('Historical baseline (physics)', 'original baseline recording; world25'),
              ('Fresh policy rollout (physics)', 'fresh exported actions and states; world28')]
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame[:62] = (28, 28, 28)
        for panel, (title, subtitle) in enumerate(titles):
            x = panel * 640 + 10
            cv2.putText(frame, title, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, f'{frames / 20:.2f}s  {subtitle}', (x, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (190, 190, 190), 1, cv2.LINE_AA)
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    if frames != 151:
        raise ValueError('Header pass lost source frames')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(raw),
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(destination)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--component', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--ego', type=Path, default=EGO)
    p.add_argument('--camera', type=Path, default=CAMERA)
    args = p.parse_args()
    base = args.base.resolve()
    component = (args.component or base / 'common_components/episode_000037').resolve()
    output = (args.output or base / 'rendered_fresh/epi37_fresh.mp4').resolve()
    if output.exists() or output.with_suffix('.video_evidence.json').exists():
        raise FileExistsError('Preserve existing fresh render: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['FLASH_CHORD_CACHE_DIR'] = str(output.parent / '_cache/assets')
    os.environ['XDG_CACHE_HOME'] = str(output.parent / '_cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('PYGLET_HEADLESS', 'true')
    import numpy as np

    source = base / 'source/episode_000037'
    source_manifest = read(base / 'source_manifest.json')
    completion = read(component / 'completed.json')
    report = completion['metrics']
    actions = Path(completion['actions']['path'])
    if sha(actions) != completion['actions']['sha256']:
        raise ValueError('Completed fresh action recording changed')
    if not report['selected_world_strict_proxy_pass'] or report['selected_world_index'] != 28 or report['world_count'] != 32:
        raise ValueError('Expected successful preselected world28 of32')
    original_recording = source / 'demo/epi37_parts/rollouts.npz'
    with np.load(actions, allow_pickle=False) as z:
        fresh = z['body_q'].copy()
        times = z['time_s'].copy()
        if not np.array_equal(z['object_body_local_ids'], [52]):
            raise ValueError('Spoon local body mapping changed')
        if not np.array_equal(fresh[:, 52:53], z['achieved_object_pose_xyzw']):
            raise ValueError('Fresh visual spoon poses differ from exported physical object poses')
    if fresh.shape != (151, 54, 7) or fresh.dtype != np.float32 or not np.isfinite(fresh).all():
        raise ValueError('Invalid fresh recorded body poses')
    if not np.allclose(times, np.arange(151) / 20., atol=1e-9, rtol=0):
        raise ValueError('Fresh record time grid differs')
    with np.load(original_recording, allow_pickle=False) as z:
        baseline, errors_baseline = z['baseline'].copy(), z['err_baseline'].copy()
        original_refined = z['refined'].copy()
        baseline_world = int(z['world_baseline'])
    if baseline.shape != fresh.shape or baseline_world != 25:
        raise ValueError('Historical baseline shape/world mismatch')
    renderer = source / 'code/render_demo.py'
    renderer_text = renderer.read_text()
    # Verify the frozen consumer branch that uses the provided recorded states.
    required_consumer = ["_r = np.load(TMP / 'rollouts.npz')", "render(PANEL, recorded(refined))",
                         "state.body_q.assign(body_q[i].astype(np.float32))",
                         "if PANEL is None and not (TMP / 'rollouts.npz').exists():"]
    if not all(text in renderer_text for text in required_consumer):
        raise ValueError('Frozen renderer no longer has the audited recorded-body consumer')
    reference = source / source_manifest['selected_reference']
    policy = source / source_manifest['selected_checkpoint']
    if sha(policy) != report['checkpoint_sha256'] or sha(reference) != report['reference_refinement_sha256']:
        raise ValueError('Renderer policy/reference differs from fresh execution')
    raw_video = output.with_name(output.stem + '_source_render.mp4')
    parts = raw_video.parent / (raw_video.stem + '_parts')
    if parts.exists():
        raise FileExistsError('Preserve existing render preparation: ' + str(parts))
    parts.mkdir()
    errors_refined = np.asarray(report['historical_demo_comparison_frames_1_through_150']['position_mean_m_per_world_object'])[:, 0] * 100
    np.savez(parts / 'rollouts.npz', refined=fresh, baseline=baseline,
             err_refined=errors_refined, err_baseline=errors_baseline,
             world_refined=28, world_baseline=25)
    preloaded = parts / 'rollouts.npz'
    preloaded_sha = sha(preloaded)
    with np.load(preloaded, allow_pickle=False) as z:
        if not np.array_equal(z['refined'], fresh) or not np.array_equal(z['baseline'], baseline):
            raise ValueError('Renderer preparation changed physical states')
    command = [sys.executable, str(renderer), str(source), str(reference), str(policy),
               str(args.ego.resolve()), str(raw_video), str(args.camera.resolve())]
    print('Rendering prerecorded fresh states; no physics rollout is requested.', flush=True)
    subprocess.run(command, check=True, cwd='/workspace')
    if sha(preloaded) != preloaded_sha:
        raise ValueError('Renderer changed its preloaded body-state recording')
    raw_validation = count_video(raw_video)
    panel_validation = {name: count_video(parts / (name + '.mp4')) for name in
                        ('kinematic_final', 'kinematic_original_hands', 'baseline', 'refined')}
    label_panels(raw_video, output, parts)
    final_validation = count_video(output)
    evidence = dict(
        schema='fresh_recorded_body_video_binding_v1',
        video_path=str(output), video_sha256=sha(output), actions_path=str(actions), actions_sha256=sha(actions),
        rendered_from_fresh_recording=True, physics_rerun_during_render=False,
        selected_world=28, baseline_world=25, frames=151, frame_count=151, fps=20.,
        fresh_body_shape=list(fresh.shape), fresh_body_dtype=str(fresh.dtype),
        fresh_body_array_sha256=array_sha(fresh), body_q_sha256=array_sha(fresh), renderer_refined_body_array_sha256=array_sha(fresh),
        renderer_input=str(preloaded), renderer_input_sha256=preloaded_sha,
        source_actions_to_renderer_input_array_equal=True,
        fresh_spoon_render_poses_equal_exported_achieved_object_poses=True,
        renderer_consumer={'file': str(renderer), 'sha256': sha(renderer),
                           'loading': "refined = np.load(TMP / 'rollouts.npz')['refined']",
                           'panel': "render('refined', recorded(refined))",
                           'frame_assignment': 'state.body_q.assign(body_q[i].astype(np.float32))',
                           'frame_range': [0, 150], 'render_time_s': 'i/20',
                           'physics_skipped_because_preloaded_rollouts_exist': True},
        renderer_command=command, bridge_script=str(Path(__file__).resolve()), bridge_sha256=sha(__file__),
        source_manifest_sha256=sha(base / 'source_manifest.json'),
        completed_execution_sha256=sha(component / 'completed.json'),
        common_build_commit=completion['code_commit_url'], common_build_manifest_sha256=completion['build_manifest_sha256'],
        original_user_video_sha256=source_manifest['selected_video_sha256'],
        original_video_sha256=source_manifest['selected_video_sha256'],
        original_user_video_role='Historical selected run demonstration; fresh states differ and are not claimed identical.',
        original_baseline_recording_sha256=sha(original_recording),
        historical_baseline_body_array_sha256=array_sha(baseline),
        fresh_vs_original_comparison=report['original_demo_comparison'],
        original_refined_body_array_sha256=array_sha(original_refined),
        policy_sha256=sha(policy), refined_reference_sha256=sha(reference),
        ego_video={'path': str(args.ego), 'sha256': sha(args.ego)},
        camera_motion={'path': str(args.camera), 'sha256': sha(args.camera)},
        panel_meanings=['Original ego video', 'Refined kinematic reference with original-hand ghost',
                        'Historical original baseline physical body recording, world25',
                        'Fresh exported physical body recording, world28'],
        postprocessing='Only the62-pixel header is relabeled; frames remain in source order at20Hz, then H264 encoded.',
        source_render_video={'path': str(raw_video), 'sha256': sha(raw_video), **raw_validation},
        panel_validation=panel_validation, final_video_validation=final_validation,
        source_reference_proxy_not_official_ground_truth=True, uploaded=False)
    evidence_path = output.with_suffix('.video_evidence.json')
    write(evidence_path, evidence)
    print(json.dumps({'video': str(output), 'evidence': str(evidence_path), 'frames': 151,
                      'rendered_from_fresh_recording': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
