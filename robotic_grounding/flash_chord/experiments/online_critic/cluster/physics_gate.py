"""Exercise prepared GPU physics without a learner, replay, or SAC compilation.

Run separately for one and 2,048 worlds, with no hand contacts and then with
production contacts/zero residual actions. This is an initialization gate, not a
training benchmark or a guarantee about later grasp contact capacity.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import warp as wp


@wp.kernel
def audit_substep(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                  nacon: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),
                  contact_capacity: int, constraint_capacity: int, worlds: int,
                  counters: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i == 0:
        wp.atomic_max(counters, 0, nacon[0])
        if nacon[0] >= contact_capacity:
            wp.atomic_add(counters, 2, 1)
    if i < worlds:
        wp.atomic_max(counters, 1, nefc[i])
        if nefc[i] >= constraint_capacity:
            wp.atomic_add(counters, 3, 1)
    finite = True
    for k in range(7):
        finite = finite and wp.isfinite(body_q[i][k])
    for k in range(6):
        finite = finite and wp.isfinite(body_qd[i][k])
    if not finite:
        wp.atomic_add(counters, 4, 1)


@wp.kernel
def gather_object_poses(body_q: wp.array(dtype=wp.transform), ids: wp.array(dtype=wp.int32),
                        poses: wp.array(dtype=wp.transform)):
    i = wp.tid()
    poses[i] = body_q[ids[i]]


@wp.kernel
def sample_contact_forces(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
                          shape1: wp.array(dtype=wp.int32), shape_body: wp.array(dtype=wp.int32),
                          force: wp.array(dtype=wp.spatial_vector), support_body: int,
                          bodies_per_world: int, maxima: wp.array(dtype=wp.float32),
                          support_net: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    if i >= count[0]:
        return
    b0 = int(-1)
    b1 = int(-1)
    if shape0[i] >= 0:
        b0 = shape_body[shape0[i]]
    if shape1[i] >= 0:
        b1 = shape_body[shape1[i]]
    f = wp.vec3(force[i][0], force[i][1], force[i][2])
    magnitude = wp.length(f)
    wp.atomic_max(maxima, 0, magnitude)
    s0 = b0 < 0 or (support_body >= 0 and b0 % bodies_per_world == support_body)
    s1 = b1 < 0 or (support_body >= 0 and b1 % bodies_per_world == support_body)
    if s0 != s1:
        wp.atomic_max(maxima, 1, magnitude)
        body = b0
        if s0:
            body = b1
            f = -f
        world = body // bodies_per_world
        for axis in range(3):
            wp.atomic_add(support_net, world, axis, f[axis])


def pose_errors(actual, target_position, target_quaternion_xyzw):
    quaternion = actual[..., 3:]
    dots = np.abs(np.sum(quaternion*target_quaternion_xyzw, axis=-1))
    norm = np.linalg.norm(quaternion, axis=-1)*np.linalg.norm(target_quaternion_xyzw, axis=-1)
    angle = 2*np.arccos(np.clip(dots/np.maximum(norm, 1e-30), 0., 1.))
    return np.linalg.norm(actual[..., :3]-target_position, axis=-1), angle


def summarize_errors(position, orientation, fps, persistence=3):
    """Errors are (post-step sample, world, object), scored against its command frame."""
    if position.shape != orientation.shape or position.ndim != 3:
        raise ValueError('Expected matching (sample, world, object) error arrays')
    finite = bool(np.isfinite(position).all() and np.isfinite(orientation).all())
    bad = (position > .05) | (orientation > np.deg2rad(20.))
    streak = np.zeros(bad.shape[1:], dtype=int)
    onset = np.full(bad.shape[1:], -1, dtype=int)
    for frame in range(len(bad)):
        streak = np.where(bad[frame], streak+1, 0)
        first = (onset < 0) & (streak >= persistence)
        onset[first] = frame-persistence+1
    objects = []
    for obj in range(position.shape[2]):
        failures = onset[:, obj][onset[:, obj] >= 0]
        start = int(failures.min()) if failures.size else None
        objects.append(dict(max_position_error_m=float(position[..., obj].max()),
            max_orientation_error_rad=float(orientation[..., obj].max()),
            failed_worlds=int(len(failures)), first_sustained_failure_command_frame=start,
            first_sustained_failure_physical_seconds=None if start is None else (start+1)/fps,
            failure_confirmed_physical_seconds=None if start is None else (start+persistence)/fps))
    return dict(passed=finite and not bool((onset >= 0).any()), finite=finite,
        thresholds=dict(position_m=.05, orientation_degrees=20., persistence_samples=persistence),
        objects=objects)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episode', default='episode_000001')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--worlds', type=int, choices=(1, 2048), required=True)
    parser.add_argument('--mode', choices=('no_hand_contacts', 'reference_control'), required=True)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error('--seconds must be finite and positive')
    root, output = args.run_root.resolve(), args.output.resolve()
    if output.exists():
        parser.error('--output must be a fresh diagnostic directory')
    subprocess.run([sys.executable, str(root/'cluster/check_source.py')], check=True)
    config_path = root/args.episode/'runs/smoke'/args.episode/'chunk'/f'seed_{args.seed}'/'launch_resolved.yaml'
    from omegaconf import OmegaConf
    import flash_chord
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import apply_evaluation_composition
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    runtime = Path(flash_chord.__file__).resolve().parents[2]
    for relative, checksum in json.loads((root/'source_manifest.json').read_text()).items():
        if digest(runtime/relative) != checksum:
            raise ValueError(f'Imported runtime differs from frozen source: {relative}')
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    original = deepcopy(config)
    apply_evaluation_composition(config, world_count=args.worlds, reset_mode='explicit')
    if args.mode == 'no_hand_contacts':
        config['collision']['robot_scope'] = 'none'
    cfg = OmegaConf.create(config)
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError('This gate requires the actual CUDA physics backend')
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output/'effective_config.yaml')
    wp.config.kernel_cache_dir = str(output/'warp_cache')
    import os
    os.environ['FLASH_CHORD_CACHE_DIR'] = str(output/'mesh_cache')
    report = dict(passed=False, mode=args.mode, worlds=args.worlds, seconds=args.seconds,
        device=str(device), device_name=device.name, config_sha256=digest(config_path),
        source_manifest_sha256=digest(root/'source_manifest.json'),
        data_manifest_sha256=digest(root/'data_manifest.json'),
        limitations='No learner/replay; physics-only memory and timings do not certify full-training memory or later contact capacity.',
        diagnostic_changes=dict(auto_reset=False, tracking_early_stop=False, assistance=0.,
                                robot_collision_disabled=args.mode == 'no_hand_contacts'))
    started = time.perf_counter()
    try:
        memory_before = int(device.free_memory)
        with wp.ScopedDevice(device):
            env_config = instantiate_typed(cfg.env, RLEnvConfig)
            kwargs = dict(parquet=cfg.task.parquet, control_fps=env_config.sim.fps,
                motion_speed=cfg.task.motion_speed,
                embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=args.worlds,
                include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                object_scale_min=cfg.scene.get('object_scale_min', 1.),
                object_scale_max=cfg.scene.get('object_scale_max', 1.),
                object_scale_seed=cfg.scene.get('object_scale_seed', 0),
                contact_friction=cfg.scene.get('contact_friction', 1.),
                object_free_joint_damping=cfg.scene.get('object_free_joint_damping', 0.),
                source_frame_playback=bool(cfg.task.get('source_frame_playback', False)),
                motion_start_frame=cfg.task.get('motion_start_frame', 0),
                motion_end_frame=cfg.task.get('motion_end_frame', -1))
            if cfg.scene.get('object_initial_pose_overrides') is not None:
                kwargs['object_initial_pose_overrides'] = OmegaConf.to_container(cfg.scene.object_initial_pose_overrides)
            setup = setup_scene(**kwargs)
            steps = round(args.seconds*setup.reference.fps)
            if not 0 < steps < setup.reference.num_frames:
                raise ValueError('Gate must sample at least one step and end before reference truncation')
            env = RLEnv(setup.scene, setup.reference, config=env_config, device=device)
            data = env.solver.mjw_data
            counters = wp.zeros(5, dtype=wp.int32)
            original_step = env.solver.step

            def monitored_step(state_in, state_out, control, contacts, dt):
                original_step(state_in, state_out, control, contacts, dt)
                wp.launch(audit_substep, dim=env.model.body_count,
                    inputs=[state_out.body_q, state_out.body_qd, data.nacon, data.nefc,
                            data.naconmax, data.njmax, args.worlds, counters])

            env.solver.step = monitored_step
            env.reset(frame_id=0)
            env.capture_step()
            env.reset(frame_id=0)
            counters.zero_()
            ids = env.command.body_ids_w
            poses = wp.empty(ids.shape[0], dtype=wp.transform)
            maxima = wp.zeros(2, dtype=wp.float32)
            support_net = wp.zeros((args.worlds, 3), dtype=wp.float32)
            wp.launch(gather_object_poses, dim=ids.shape[0], inputs=[env.state_0.body_q, ids, poses])
            initial = poses.numpy().reshape(args.worlds, -1, 7)
            initial_position, initial_orientation = pose_errors(initial,
                np.asarray(setup.reference.object_body_pos_w())[0],
                np.asarray(setup.reference.object_body_quat_w())[0][..., [1, 2, 3, 0]])
            initial_velocity = float(np.abs(env.state_0.body_qd.numpy()[ids.numpy()]).max())
            initial_passed = bool(initial_position.max() <= .05
                and initial_orientation.max() <= np.deg2rad(20.) and initial_velocity <= 1e-8)
            report['initial_state'] = dict(passed=initial_passed, poses_xyzw_world0=initial[0].tolist(),
                maximum_position_error_per_object_m=initial_position.max(axis=0).tolist(),
                maximum_orientation_error_per_object_degrees=np.rad2deg(initial_orientation.max(axis=0)).tolist(),
                object_pose_overrides=config['scene'].get('object_initial_pose_overrides'),
                max_object_velocity_m_s_or_rad_s=initial_velocity)
            wp.synchronize_device(device)
            report['build_and_capture_seconds'] = time.perf_counter()-started
            trace, frames, step_times, memory = [], [], [], [memory_before, int(device.free_memory)]
            force_trace = []
            for step in range(steps):
                frame = env.timestep.numpy().copy()
                before = time.perf_counter()
                env.step()  # Persistent action_input is zero: no residual policy correction.
                wp.synchronize_device(device)
                step_times.append(time.perf_counter()-before)
                wp.launch(gather_object_poses, dim=ids.shape[0], inputs=[env.state_0.body_q, ids, poses])
                trace.append(poses.numpy().reshape(args.worlds, -1, 7))
                frames.append(frame)
                memory.append(int(device.free_memory))
                maxima.zero_()
                support_net.zero_()
                wp.launch(sample_contact_forces, dim=env.contacts.rigid_contact_max,
                    inputs=[env.contacts.rigid_contact_count, env.contacts.rigid_contact_shape0,
                        env.contacts.rigid_contact_shape1, env.model.shape_body, env.contacts.force,
                        -1 if setup.scene.support is None else setup.scene.support.body_id,
                        env.model.body_count//args.worlds, maxima, support_net])
                force_max = maxima.numpy()
                force_trace.append(dict(physical_seconds=(step+1)/setup.reference.fps,
                    maximum_contact_force_N=float(force_max[0]),
                    maximum_support_contact_force_N=float(force_max[1]),
                    support_net_force_on_dynamic_bodies_world0_N=support_net.numpy()[0].tolist()))
            trace, frames = np.asarray(trace), np.asarray(frames)
            target_p = np.asarray(setup.reference.object_body_pos_w())[frames]
            target_q = np.asarray(setup.reference.object_body_quat_w())[frames][..., [1, 2, 3, 0]]
            position, orientation = pose_errors(trace, target_p, target_q)
            quality = summarize_errors(position, orientation, setup.reference.fps)
            early_steps = min(steps, round(.5*setup.reference.fps))
            early_quality = summarize_errors(position[:early_steps], orientation[:early_steps], setup.reference.fps)
            required_quality = quality if args.mode == 'no_hand_contacts' else early_quality
            sequential_frames = bool(np.array_equal(frames,
                np.broadcast_to(np.arange(steps)[:, None], (steps, args.worlds))))
            health = counters.numpy().tolist()
            report.update(object_names=setup.reference.object_body_names(), quality=quality,
                first_half_second_quality=early_quality,
                required_pose_horizon_seconds=args.seconds if args.mode == 'no_hand_contacts' else .5,
                contact_force_samples=force_trace,
                force_sampling='Last physics substep of each control frame; does not bound impulses between samples.',
                configured_nconmax_per_world=original['sim']['nconmax'],
                configured_njmax_per_world=original['sim']['njmax'],
                actual_contact_capacity_all_worlds=data.naconmax,
                actual_constraint_capacity_per_world=data.njmax,
                sequential_reference_frames=sequential_frames,
                max_contacts_all_worlds=health[0], max_constraints_per_world=health[1],
                contact_capacity_saturated_substeps=health[2],
                constraint_capacity_saturated_world_substeps=health[3], nonfinite_body_substeps=health[4],
                physics_step_seconds=step_times,
                measured_transitions_per_second=args.worlds*steps/sum(step_times),
                device_free_bytes_before=memory_before, device_free_bytes_min=min(memory),
                device_total_bytes=int(device.total_memory),
                host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                passed=initial_passed and sequential_frames and required_quality['passed'] and health[2:] == [0, 0, 0])
            np.savez_compressed(output/'trace.npz', poses=trace, command_frames=frames,
                physical_times_s=(np.arange(steps)+1)/setup.reference.fps,
                position_error_m=position, orientation_error_rad=orientation)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['total_wall_seconds'] = time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
