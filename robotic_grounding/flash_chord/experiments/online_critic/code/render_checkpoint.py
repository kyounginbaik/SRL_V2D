"""Render matched unassisted and training-VOC checkpoint previews."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent.parent
phase, ep, variant, seed, step = sys.argv[1:6]
step = int(step)
mode = sys.argv[6] if len(sys.argv) > 6 else 'both'
if mode not in ('both', 'voc', 'unassisted'):
    raise ValueError(f'Unknown render mode: {mode}')
run = BASE / 'runs' / phase / ep / variant / f'seed_{seed}'
checkpoint = run / 'training' / f'policy_{step}.safetensors'
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
reference = pq.read_table(BASE / 'frozen' / ep / 'reference.parquet').to_pylist()[0]
reference_fps = float(reference['fps'])
if not np.isfinite(reference_fps) or reference_fps <= 0:
    raise ValueError(f'Invalid reference frame rate: {reference_fps}')
reference_frames = len(reference['object_body_position'])
positions = np.asarray(reference['object_body_position']).reshape(-1, 3)
center = np.median(positions, axis=0)
radius = max(.5, float(np.percentile(np.linalg.norm(positions - center, axis=1), 95)) + .35)
camera = center + radius * np.array([1.8, -2.3, 1.7])


def training_voc_scale():
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    meta = read_checkpoint_metadata(checkpoint)
    if int(meta['environment_steps']) != step:
        raise ValueError('Checkpoint step mismatch')
    return float(meta['voc_scale'])


def validate_video(path):
    import cv2
    if not path.is_file() or path.stat().st_size <= 1000:
        raise RuntimeError(f'Missing or empty rendered video: {path}')
    cap = cv2.VideoCapture(str(path))
    try:
        frames = 0
        while cap.read()[0]:
            frames += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if frames != reference_frames or not np.isfinite(fps) or abs(fps-reference_fps) > 1e-4:
        raise ValueError(f'Incomplete rollout video {path}: {frames} frames, {fps} fps; '
                         f'expected {reference_frames} frames, {reference_fps} fps')
    return frames, size


def checkpoint_estimates(recording, output, scale):
    """Optional local proxy scores; unsupported symmetry never invents an axis."""
    unsupported = sorted(set(reference['object_body_names']) & {'beige_cup', 'blue_cup'})
    if not unsupported:
        from flash_chord.evaluation.track3_estimates import score_recording
        score_recording(recording, BASE / 'frozen' / ep / 'reference.parquet', output)
        return {'available': True}
    # Keep validating the required physical rollout even when proxy geometry
    # cannot supply a justified symmetry-aware metric for this episode.
    with np.load(recording, allow_pickle=False) as data:
        poses = data['achieved_pose_w']
        expected_shape = (reference_frames, 1, len(reference['object_body_names']), 7)
        if (not np.array_equal(data['frame_indices'], np.arange(reference_frames))
                or poses.shape != expected_shape or not np.isfinite(poses).all()
                or np.any(np.linalg.norm(poses[..., 3:], axis=-1) < 1e-8)):
            raise ValueError('Checkpoint recording must contain valid poses for every reference frame')
        if not np.isclose(float(data['voc_scale']), scale):
            raise ValueError('Checkpoint recording has the wrong VOC scale')
    status = {'available': False, 'reason': 'Verified cup symmetry axis unavailable for local proxy metrics',
              'unsupported_objects': unsupported}
    output.write_text(json.dumps(dict(status, metrics={}, estimated=True, voc_scale=scale,
        frames=reference_frames, worlds=1,
        reference_parquet=str(BASE / 'frozen' / ep / 'reference.parquet')), indent=2)+'\n')
    print(f'[checkpoint-media] Optional Track 3 proxy metrics unavailable for {ep}: '
          f'{status["reason"]} ({", ".join(unsupported)})', flush=True)
    return status


def render(assisted):
    scale = training_voc_scale() if assisted else 0.0
    folder = 'voc_checkpoint_videos' if assisted else 'checkpoint_videos'
    out = run / folder / f'step_{step}'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{ep}.mp4'
    complete = out / f'{ep}.complete'
    if complete.is_file() and path.is_file() and path.stat().st_size > 1000:
        print(f'Existing render: {path}', flush=True)
        return
    args = [
        'scripts/view_policy.py', f'evaluation.checkpoint={checkpoint}',
        'evaluation.reset_mode=explicit', 'scene.world_count=1', 'training.world_count=1',
        'env.auto_reset=false', f'env.voc_scale={scale}', 'reset.voc_decay_steps=0',
        'reset.always_reset_to_first_frame=true',
        *[f'termination.{term}.enabled=false' for term in
          ('wrist_position', 'wrist_orientation', 'object_position', 'object_orientation')],
        'viewer.backend=mp4', f'viewer.output_path={path}',
        f'viewer.num_frames={reference_frames}',
        f'viewer.video_fps={reference_fps}', 'viewer.width=960', 'viewer.height=540',
        'viewer.headless=true', 'viewer.realtime=false', 'markers.enabled=false',
        'viewer.camera_position=' + json.dumps(camera.tolist(), separators=(',', ':')),
        'viewer.camera_target=' + json.dumps(center.tolist(), separators=(',', ':')),
        f'hydra.run.dir={out / "hydra" / ep}',
    ]
    if os.environ.get('V2D_RENDER_CPU') == '1':
        args.append('training.device=cpu')
    if scale == 0.0:
        args += ['env.voc.max_force=0.0', 'env.voc.max_torque=0.0']
    recording = out / 'object_poses.npz'
    subprocess.run([sys.executable, *args], cwd='/workspace', check=True,
                   env={**os.environ, 'UNIRETARGET_POSE_RECORDING': str(recording)})
    estimates = checkpoint_estimates(recording, out / 'track3_estimates.json', scale)
    import cv2, imageio_ffmpeg
    frames, size = validate_video(path)
    raw=out / 'raw.mp4';path.replace(raw)
    cap=cv2.VideoCapture(str(raw))
    intermediate=out/'labelled_raw.mp4'
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),reference_fps,size)
    assert writer.isOpened()
    label=f"Step {step} | VOC {scale:g} | " + ('training assistance' if assisted else 'unassisted eval')
    for _ in range(frames):
        ok,im=cap.read();assert ok
        cv2.rectangle(im,(0,0),(size[0],32),(0,0,0),-1)
        cv2.putText(im,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        writer.write(im)
    cap.release();writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(intermediate),
        '-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
    validate_video(path)
    (out / 'manifest.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'environment_steps': step,
        'voc_scale': scale, 'assisted': assisted,
        'full_horizon': True, 'reset_assist_steps': 0,
        'object_force_caps': 'checkpoint_config' if scale > 0 else 'zero',
        'video_role': 'training_voc_setting' if assisted else 'unassisted_evaluation',
        'termination_disabled_for_full_horizon_diagnostic': True,
        'frames': reference_frames, 'fps': reference_fps,
        'track3_proxy_metrics': estimates,
    }, indent=2) + '\n')
    # The encoded output and its manifest are durable before disposable
    # intermediates are removed. Failures above retain raw evidence for repair.
    raw.unlink()
    intermediate.unlink()
    complete.write_text('Rendered checkpoint successfully\n')
    print(path, flush=True)


if mode in ('both', 'unassisted'):
    render(False)
if mode in ('both', 'voc'):
    render(True)
