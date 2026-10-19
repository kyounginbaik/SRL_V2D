"""Four simultaneous, unassisted saved-policy rollouts with synchronized evidence.

This is checkpoint evaluation, not captured learner training. Record each state
before its action so output frame i corresponds exactly to reference frame i.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finalize_archive(args, base, run, out, path, complete):
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['full_horizon'] and not args.skip_archive and not manifest.get('success_archive_evaluated',False):
        archive_script=base/'code/archive_success.py'
        if not archive_script.exists():raise FileNotFoundError(archive_script)
        subprocess.run([sys.executable,str(archive_script),'--trajectory',str(out/'rollout.npz'),'--manifest',str(manifest_path),'--archive-root',str(args.archive_root or run/'success_archive'),'--video',str(path)],check=True)
        manifest['success_archive_evaluated']=True
    manifest.setdefault('success_archive_evaluated',False)
    manifest['archive_status']='evaluated' if manifest['success_archive_evaluated'] else 'pending_or_partial_diagnostic'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    complete.write_text('Parallel checkpoint rollout and evidence export complete; success archive '+manifest['archive_status']+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase'); parser.add_argument('episode'); parser.add_argument('variant'); parser.add_argument('seed', type=int); parser.add_argument('step', type=int)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--skip-archive', action='store_true')
    args = parser.parse_args()
    base = args.root
    run = base/'runs'/args.phase/args.episode/args.variant/f'seed_{args.seed}'
    checkpoint = args.checkpoint or run/'training'/f'policy_{args.step}.safetensors'
    out = args.output or run/'parallel_checkpoint_videos'/f'step_{args.step}'
    out.mkdir(parents=True, exist_ok=True)
    path = out/f'{args.episode}.mp4'
    complete = path.with_suffix('.complete')
    reference_path = base/'frozen'/args.episode/'reference.parquet'
    row = pq.read_table(reference_path).to_pylist()[0]
    full_frames = len(row['object_body_position'])
    frames = full_frames if args.max_frames is None else min(args.max_frames, full_frames)
    if frames < 1 or args.worlds < 1:
        raise ValueError('Positive frames and worlds required')
    fps = float(row['fps'])
    names = list(row['safe_object_body_names'])
    if path.is_file() and (out/'rollout.npz').is_file() and (out/'manifest.json').is_file():
        previous=json.loads((out/'manifest.json').read_text())
        if previous['frames']==frames and previous['world_count']==args.worlds and previous['checkpoint_sha256']==digest(checkpoint) and previous['source_reference_sha256']==digest(reference_path):
            finalize_archive(args,base,run,out,path,complete)
            print(f'Existing parallel checkpoint rollout finalized without rerender: {path}',flush=True)
            return
    reference = np.concatenate([np.asarray(row['object_body_position']), np.asarray(row['object_body_wxyz'])[..., [1,2,3,0]]], axis=-1).astype(np.float32)

    import jax.numpy as jnp
    import warp as wp
    from hydra import compose, initialize_config_dir
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_evaluation_config
    from flash_chord.visualization.viewer import ViewerConfig, create_viewer, configure_camera
    metadata = read_checkpoint_metadata(checkpoint)
    saved = checkpoint_config(metadata)
    overrides = [
        f'evaluation.checkpoint={checkpoint}', f'task.parquet={reference_path}', 'scene.support=true', 'evaluation.reset_mode=explicit',
        'evaluation.start_frame=0', 'evaluation.deterministic=false',
        f'scene.world_count={args.worlds}', f'training.world_count={args.worlds}',
        f'training.seed={args.seed}', 'env.auto_reset=false', 'env.voc_scale=0.0',
        'env.voc.max_force=0.0', 'env.voc.max_torque=0.0',
        'reset.voc_decay_steps=0', 'reset.always_reset_to_first_frame=true',
        *[f'termination.{term}.enabled=false' for term in ('wrist_position','wrist_orientation','object_position','object_orientation','reference_end')],
        'viewer.backend=null', 'viewer.headless=true', 'viewer.realtime=false', 'markers.enabled=false',
    ]
    if os.environ.get('V2D_RENDER_CPU') == '1':
        overrides.append('training.device=cpu')
    with initialize_config_dir(version_base='1.3', config_dir='/workspace/src/flash_chord/configs'):
        requested = compose(config_name='view_policy', overrides=overrides)
    cfg = checkpoint_evaluation_config(requested, saved, tuple(x.split('=',1)[0] for x in overrides), use_evaluation_training_world_count=False)
    # The viewer application supplies the same actor loader, reset and physics as
    # single-world checkpoint previews. Its step wrapper can explicitly reset at
    # episode ends, so this evidence loop directly advances its JAX environment.
    spec = importlib.util.spec_from_file_location('parallel_policy_view', '/workspace/scripts/view_policy.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    raw = out/'parallel_raw.mp4'
    viewer_config = ViewerConfig(backend='mp4', output_path=str(raw), num_frames=frames, video_fps=fps, width=1280, height=960, headless=True, realtime=False)
    with wp.ScopedDevice(cfg.training.device):
        viewer = create_viewer(viewer_config, device=cfg.training.device)
        try:
            app = module.FlashSACPolicyViewer(viewer, SimpleNamespace(config=cfg))
            env = app.env
            if env.config.auto_reset or env.reset_policy.config.voc_decay_steps != 0:
                raise ValueError('Unassisted evidence requires no resets or settling assistance')
            if not np.all(env.voc_scale.numpy() == 0):
                raise ValueError('VOC must be zero')
            viewer.set_world_offsets((1.10,1.10,0.0))
            offsets = viewer.world_offsets.numpy()
            object_center = np.median(np.asarray(row['object_body_position']).reshape(-1,3),axis=0)
            target = object_center + offsets.mean(axis=0)
            span = max(1.1, float(np.ptp(offsets,axis=0).max()) + 1.1)
            camera = target + span*np.array([.9,-1.15,1.30])
            configure_camera(viewer, ViewerConfig(camera_position=tuple(camera), camera_target=tuple(target)))
            projection = np.asarray(viewer.camera.get_projection_matrix()).reshape(4,4).T @ np.asarray(viewer.camera.get_view_matrix()).reshape(4,4).T
            label_points = object_center[None,:] + offsets + np.array([0.,0.,.30])
            label_clip = np.concatenate([label_points,np.ones((args.worlds,1))],axis=1) @ projection.T
            label_pixels = (label_clip[:,:2]/label_clip[:,3:4]*np.array([1.,-1.])+1)*np.array([viewer_config.width,viewer_config.height])/2
            bodies_per_world = app.scene.model.body_count // args.worlds
            bindings = sorted(app.scene.objects, key=lambda x:x.root.reference_body_id)
            if [x.root.reference_body_id for x in bindings] != list(range(len(names))):
                raise ValueError('Object root order does not match canonical reference bodies')
            body_ids = [x.root.body_id for x in bindings]
            actual=[]; joint_q=[]; actions=[]; frame_indices=[]; reset_count=0
            exploded_worlds=np.zeros(args.worlds,bool)
            for frame in range(frames):
                timesteps=env.timestep.numpy()
                if not np.all(timesteps == frame):
                    raise ValueError(f'Reference synchronization failed at frame {frame}: {timesteps.tolist()}')
                body_q=env.state_0.body_q.numpy().reshape(args.worlds,bodies_per_world,7)
                pose=body_q[:,body_ids].copy()
                if not np.isfinite(pose).all():
                    bad=~np.isfinite(pose).all(axis=(1,2))
                    if frame==0 or not actual: raise FloatingPointError(f'Nonfinite object pose at frame {frame}')
                    pose[bad]=actual[-1][bad]; exploded_worlds|=bad
                joints=env.state_0.joint_q.numpy().reshape(args.worlds,-1).copy()
                if not np.isfinite(joints).all():
                    badj=~np.isfinite(joints).all(axis=1)
                    if not joint_q: raise FloatingPointError(f'Nonfinite robot state at frame {frame}')
                    joints[badj]=joint_q[-1][badj]; exploded_worlds|=badj
                actual.append(pose); joint_q.append(joints); frame_indices.append(frame)
                viewer.begin_frame(frame/fps); viewer.log_state(env.state_0); viewer.end_frame()
                if frame+1 < frames:
                    action,app.key=app.policy_action(app.actor_state,app.observation,app.key)
                    action_array=np.asarray(action).copy()
                    if not np.isfinite(action_array).all():
                        exploded_worlds|=~np.isfinite(action_array).all(axis=1); action_array=np.nan_to_num(action_array); action=jnp.asarray(action_array)
                    actions.append(action_array)
                    transition=app.jax_env.step(action)
                    app.observation=jnp.array(transition.observation,copy=True)
                    reset_count += int(env.reset_mask.numpy().sum())
            if reset_count:
                raise ValueError(f'Unexpected resets during rollout: {reset_count}')
        finally:
            viewer.close()
    trajectory=out/'rollout.npz'
    np.savez_compressed(trajectory,actual=np.asarray(actual,dtype=np.float32),reference=reference[:frames],frame_indices=np.asarray(frame_indices,np.int32),actions=np.asarray(actions,dtype=np.float32),joint_q=np.asarray(joint_q,dtype=np.float32),object_names=np.asarray(names),fps=np.asarray(fps))
    import cv2
    import imageio_ffmpeg
    capture=cv2.VideoCapture(str(raw)); size=(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for i in range(frames):
        ok,image=capture.read()
        if not ok:raise ValueError(f'Parallel video ended early at frame {i}')
        cv2.rectangle(image,(0,0),(size[0],65),(15,18,24),-1)
        cv2.putText(image,f'PARALLEL CHECKPOINT ROLLOUT | {args.worlds} worlds | Step {args.step:,} | VOC 0',(16,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(image,f'Stochastic saved policy | t={i/fps:.2f}s | full continuous rollout | no resets',(16,53),cv2.FONT_HERSHEY_SIMPLEX,.56,(210,220,232),1,cv2.LINE_AA)
        for world,pixel in enumerate(label_pixels):
            x,y=np.rint(pixel).astype(int)
            if 0<=x<size[0]-90 and 85<=y<size[1]-10:
                cv2.putText(image,f'World {world}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,210,255),2,cv2.LINE_AA)
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close(); capture.release()
    if encoder.wait()!=0:raise RuntimeError('Parallel video encoding failed')
    snapshot=saved['task'].get('robot_reference_path')
    manifest={'episode':args.episode,'environment_steps':args.step,'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'source_reference_path':str(reference_path),'source_reference_sha256':digest(reference_path),'reference_snapshot_path':snapshot,'reference_snapshot_sha256':digest(snapshot) if snapshot else None,'object_names':names,'fps':fps,'frames':frames,'world_count':args.worlds,'world_seeds':[args.seed]*args.worlds,'seed':args.seed,'action_context':'Stochastic saved policy; shared batched RNG seed with distinct per-world samples. No optimizer updates.','evaluation_script_path':str(Path(__file__).resolve()),'evaluation_script_sha256':digest(__file__),'eval_code_sha256':digest('/workspace/scripts/view_policy.py'),'video_role':'parallel_checkpoint_rollout','video_path':str(path),'trajectory_path':str(trajectory),'voc_scale':0.0,'voc_force_limit':0.0,'voc_torque_limit':0.0,'reset_assist_steps':0,'full_horizon':frames==full_frames,'reset_count_after_start':reset_count,'auto_reset':False,'object_pose_forcing':False,'frame_alignment':'State recorded before action i; exact canonical frame_indices 0..T-1.','world_visual_offsets_m':offsets.tolist(),'camera_position':camera.tolist(),'camera_target':target.tolist()}
    manifest_path=out/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    finalize_archive(args,base,run,out,path,complete)
    print('PARALLEL_CHECKPOINT_COMPLETE',path,flush=True)


if __name__=='__main__':main()
