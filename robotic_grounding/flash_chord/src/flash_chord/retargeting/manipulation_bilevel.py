"""Experimental ReActor-inspired manipulation updates with simulator-scored search.

This is NOT an exact bilevel hypergradient. ReActor proposes a stopped-rollout
body-map step; a common-random-number Newton search scores the projected robot
references. The wrench arm adds CHORD's contact-wrench SUPPORT mismatch (not a
measured net force) to that search. Object and contact references stay fixed.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import newton
import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.binding import build_robot_reference_binding, _frame_poses
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action
from flash_chord.retargeting.reactor import (
    BodyMotion, NominalAlignment, ReactorLimits, map_motion, outer_step, zero_parameters,
)
from flash_chord.retargeting.reference_snapshot import save_snapshot, command_digest


def body_motion(position, quaternion_xyzw):
    position = np.asarray(position, dtype=np.float32)
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw).reshape(-1, 4)).as_matrix()
    return BodyMotion(jnp.asarray(position), jnp.asarray(rotation.reshape(*position.shape[:-1], 3, 3)),
                      jnp.zeros_like(position), jnp.zeros_like(position))


def binding_motion(binding):
    positions, quats = [], []
    for hand in binding.hands:
        positions.append(hand.keypoint_pos_w("fingertip"))
        q = np.concatenate((hand.wrist_quat_w[:, None], hand.fingertip_quat_w), axis=1)
        quats.append(q[..., [1, 2, 3, 0]])
    return body_motion(np.concatenate(positions, axis=1), np.concatenate(quats, axis=1))


def candidate_score(metrics, wrench_weight):
    # Dimensionless terms with fixed, declared scales; identical in both arms.
    return metrics["robot_position_mse"] / 0.02**2 + 0.1 * metrics["robot_rotation_mse"] / 0.2**2 + wrench_weight * metrics["wrench_mse"]


def passes_task_gate(candidate, baseline, tolerance_m=0.002):
    return (all(np.all(np.isfinite(v)) for v in candidate.values())
            and candidate["object_position_mean_m"] <= baseline["object_position_mean_m"] + tolerance_m
            and candidate["object_position_p95_m"] <= baseline["object_position_p95_m"] + tolerance_m
            and candidate["object_orientation_mean_rad"] <= baseline["object_orientation_mean_rad"] + 0.01
            and candidate["object_success_fraction"] >= baseline["object_success_fraction"] - 0.01)


def assert_device_binding(env, binding):
    np.testing.assert_allclose(env.reset_table.ref_joint_q.numpy()[:, :binding.num_joint_q], binding.joint_q, atol=1e-7)
    for consumer in (env.action, env.observation_strategy):
        reference = consumer.reference if hasattr(consumer, "reference") else consumer.buffers.reference
        np.testing.assert_allclose(reference.joint_target.numpy().reshape(binding.joint_target.shape),
                                   binding.joint_target, atol=1e-7)
    keypoints = np.concatenate([binding.hand(side).keypoint_pos_w(env.objective.config.hand_keypoint_frame)
                               for side in env.objective.config.sides], axis=1)
    np.testing.assert_allclose(env.objective.ref_hand_keypoints_w.numpy().reshape(keypoints.shape), keypoints, atol=1e-7)


class LocalProjector:
    """Bounded numerical IK on the EXACT Newton robot, never the G1 model.

    Solve sparse temporal knots then smooth their joint deltas. Recompute all FK
    targets from the result. Joint limits, baseline trust region and added joint
    velocity are checked; object performance is screened in physical rollouts.
    This is not a collision-free or dynamic-feasibility certificate.
    """
    def __init__(self, embodiment, baseline, stride=8):
        self.builder = newton.ModelBuilder()
        self.layout = embodiment.build(self.builder)
        self.baseline = baseline
        scalar = self.layout.scalar_joints
        self.qids = np.asarray(scalar.q_ids)
        self.dids = np.asarray(scalar.dof_ids)
        self.ball_joints = [(self.builder.joint_q_start[j], self.builder.joint_qd_start[j])
                            for j, kind in enumerate(self.builder.joint_type)
                            if kind == newton.JointType.BALL]
        if (len(self.qids) + 4 * len(self.ball_joints) != baseline.num_joint_q
                or len(self.dids) + 3 * len(self.ball_joints) != baseline.num_joint_dof):
            raise ValueError("reference editor supports scalar joints plus ball wrists")
        self.model = self.builder.finalize(device="cpu", skip_all_validations=True)
        self.state = self.model.state()
        self.state.joint_qd.zero_()
        self.frames = tuple(f for h in self.layout.hands for f in (h.palm_frame, *h.fingertip_frames))
        self.lower = self.model.joint_limit_lower.numpy()[self.dids]
        self.upper = self.model.joint_limit_upper.numpy()[self.dids]
        self.knots = np.unique(np.r_[np.arange(0, baseline.num_frames, stride), baseline.num_frames - 1])

    def fk(self, q):
        self.state.joint_q.assign(np.asarray(q, dtype=np.float32))
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        pos, quat = _frame_poses(self.state.body_q.numpy()[None], self.frames)
        return pos[0], Rotation.from_quat(quat[0])

    def project(self, targets):
        position = np.asarray(targets.position)
        rotation = np.asarray(targets.rotation)
        deltas = []
        for t in self.knots:
            origin = self.baseline.joint_q[t].copy()
            q = origin.copy()
            target_r = Rotation.from_matrix(rotation[t])
            for _ in range(2):
                p, r = self.fk(q)
                residual = np.r_[(position[t] - p).ravel(), 0.05 * (target_r * r.inv()).as_rotvec().ravel()]
                jac = np.empty((residual.size, len(self.qids)))
                for j, qid in enumerate(self.qids):
                    shifted = q.copy()
                    shifted[qid] += 0.001
                    p1, r1 = self.fk(shifted)
                    jac[:, j] = np.r_[(p1-p).ravel(), 0.05*(r1*r.inv()).as_rotvec().ravel()] / 0.001
                step = np.linalg.solve(jac.T @ jac + 0.005 * np.eye(len(self.qids)), jac.T @ residual)
                q[self.qids] = np.clip(q[self.qids] + np.clip(step, -0.05, 0.05),
                    np.maximum(self.lower, origin[self.qids]-0.12),
                    np.minimum(self.upper, origin[self.qids]+0.12))
            deltas.append(q-origin)
        delta = np.stack([np.interp(np.arange(self.baseline.num_frames), self.knots, np.asarray(deltas)[:, j])
                          for j in range(self.baseline.num_joint_q)], axis=1)
        delta = gaussian_filter1d(delta, 2.0, axis=0)
        delta *= np.minimum(np.arange(len(delta)) / 10.0, 1.0)[:, None]
        q = self.baseline.joint_q + delta
        q[:, self.qids] = np.clip(q[:, self.qids], self.lower, self.upper)
        q[0] = self.baseline.joint_q[0]
        if np.max(np.abs(np.diff(q-self.baseline.joint_q, axis=0))) * self.baseline.fps > 0.5:
            raise ValueError("projection exceeded added joint velocity trust region 0.5 rad/s")
        target = self.baseline.joint_target.copy()
        target[:, self.dids] = q[:, self.qids]
        return build_robot_reference_binding(self.builder, self.layout, q, target, self.baseline.fps)


class ManipulationBilevel:
    def __init__(self, *, options, setup, embodiment, collision, training, environment_config,
                 checkpoint_metadata, output_dir):
        self.options = dict(options)
        self.mode = self.options.get("mode", "fixed")
        if self.mode not in ("fixed", "reactor", "reactor_wrench"):
            raise ValueError("bilevel.mode must be fixed, reactor or reactor_wrench")
        if training.resume:
            raise ValueError("bilevel resume is not yet supported; refusing to reset upper-state silently")
        self.interval = int(self.options.get("interval", 9830400))
        self.next_update = int(self.options.get("warmup", self.interval))
        if self.interval <= 0 or self.next_update <= 0:
            raise ValueError("positive upper warmup/interval required")
        self.training = training
        self.setup = setup
        self.config = deepcopy(environment_config)
        self.metadata = checkpoint_metadata
        self.output = Path(output_dir) / "references"
        self.output.mkdir(parents=True, exist_ok=True)
        self.baseline = setup.scene.robot_reference
        self.command_sha256 = command_digest(setup.reference)
        self.binding = self.baseline
        self.source = binding_motion(self.baseline)
        b = self.source.position.shape[1]
        self.theta = zero_parameters(b, 1)
        self.nominal = NominalAlignment(1.0, jnp.zeros((b,3)), jnp.broadcast_to(jnp.eye(3), (b,3,3)))
        self.clip_ids = jnp.zeros(self.baseline.num_frames, dtype=jnp.int32)
        self.heights = jnp.zeros(1)
        self.projector = None if self.mode == "fixed" else LocalProjector(embodiment, self.baseline)
        self.frames = tuple(f for h in setup.scene.layout.hands for f in (h.palm_frame, *h.fingertip_frames))
        self.version = 0
        self.validation_steps = 0
        active = np.zeros(self.baseline.num_frames, dtype=bool)
        for side in setup.reference.sides:
            active |= np.asarray(setup.reference.contact_active(side)).reshape(self.baseline.num_frames,-1).any(axis=1)
        if not active.any():
            raise ValueError("wrench ablation requires nonempty reference contact labels")
        # Balance pre-contact, contact, and remaining no-contact phases.
        self.phases = np.where(active, 1, np.where(np.arange(len(active)) < np.flatnonzero(active)[0], 0, 2))
        self.rng = np.random.default_rng(training.seed + 10000)
        evaluation = deepcopy(self.config)
        apply_evaluation_composition(evaluation, world_count=int(self.options.get("eval_worlds", 4)))
        self.eval_config = instantiate_typed(OmegaConf.create(evaluation["env"]), RLEnvConfig)
        scene = evaluation["scene"]
        task = evaluation["task"]
        self.eval_setup = setup_scene(parquet=task["parquet"], control_fps=self.eval_config.sim.fps,
            motion_speed=task["motion_speed"], embodiment=embodiment, collision=collision,
            world_count=scene["world_count"], include_support=scene["support"],
            decompose_objects=scene["decompose_objects"],
            contact_friction=scene.get("contact_friction", 1.0),
            object_free_joint_damping=scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=task["source_frame_playback"],
            motion_start_frame=task["motion_start_frame"], motion_end_frame=task["motion_end_frame"])
        self.contact_proximity = None
        if float(self.options.get("contact_proximity_weight", 0.0)) > 0:
            from flash_chord.retargeting.contact_proximity import ContactProximity
            self.contact_proximity = ContactProximity(
                self.eval_setup.reference, [hand.side for hand in self.eval_setup.scene.layout.hands]
            )
        self.actor = None
        self.policy = None
        self.publish(0)
        if self.projector is not None:
            self.validate_nonzero_binding()

    def validate_nonzero_binding(self):
        """Exercise the otherwise conditional publication path before a long run."""
        q = self.baseline.joint_q.copy()
        qid = self.projector.qids[0]
        q[:,qid] += .001 * np.sin(np.linspace(0,np.pi,len(q)))
        q[:,self.projector.qids] = np.clip(q[:,self.projector.qids],self.projector.lower,self.projector.upper)
        target = self.baseline.joint_target.copy()
        target[:,self.projector.dids] = q[:,self.projector.qids]
        binding = build_robot_reference_binding(self.projector.builder,self.projector.layout,q,target,self.baseline.fps)
        check = RLEnv(replace(self.eval_setup.scene,robot_reference=binding),self.eval_setup.reference,
                      config=deepcopy(self.eval_config))
        assert_device_binding(check,binding)
        frame = len(q)//2
        check.reset(frame_id=frame)
        actual = check.state_0.joint_q.numpy().reshape(check.world_count,-1)
        np.testing.assert_allclose(actual[:,qid],q[frame,qid],atol=1e-6)
        if np.max(np.abs(binding.joint_q-self.baseline.joint_q)) < 1e-6:
            raise AssertionError("nonzero reference wiring test did not change a joint")
        if command_digest(self.eval_setup.reference) != self.command_sha256:
            raise AssertionError("nonzero reference wiring test changed object/contact commands")
        (self.output.parent/"binding_wiring_check.json").write_text(json.dumps({
            "passed":True,"changed_joint":self.projector.layout.scalar_joints.names[0],
            "max_joint_change_rad":float(np.max(np.abs(binding.joint_q-self.baseline.joint_q))),
            "checked":["action","observation","objective","reset_table","actual_reset","object_contact_invariance"]
        },indent=2)+"\n")

    def publish(self, step):
        path = self.output / f"reference_{self.version:04d}.npz"
        checksum = save_snapshot(path, self.binding, self.setup.scene.layout,
            provenance=dict(mode=self.mode, version=self.version, environment_steps=step,
                            source_parquet=self.config["task"]["parquet"],
                            object_contact_sha256=self.command_sha256))
        with (self.output / f"theta_{self.version:04d}.npz").open("wb") as stream:
            np.savez(stream, position=np.asarray(self.theta.position), orientation=np.asarray(self.theta.orientation),
                     height=np.asarray(self.theta.height), frame_names=np.asarray([f.name for f in self.frames]))
        self.config["task"]["robot_reference_path"] = str(path)
        self.config["task"]["robot_reference_sha256"] = checksum
        self.metadata[CONFIG_METADATA_KEY] = json.dumps(self.config, sort_keys=True)
        self.metadata["robot_reference_version"] = str(self.version)

    def rollout(self, binding, learner, seed, *, return_feedback=False, voc_probe_scale=None, initial_q_delta=None):
        probe_config=deepcopy(self.eval_config)
        if voc_probe_scale is not None:
            from flash_chord.runtime.object_control import VOCParams
            probe_config=replace(probe_config,
                voc=instantiate_typed(OmegaConf.create(self.config["env"]["voc"]), VOCParams),
                voc_scale=float(voc_probe_scale))
        env = RLEnv(replace(self.eval_setup.scene, robot_reference=binding), self.eval_setup.reference,
                    config=probe_config)
        env.reset(frame_id=0)
        env.capture_step()
        env.reset(frame_id=0)
        if getattr(self, "penetration_aware", False):
            from flash_chord.retargeting.penetration import (
                HandObjectPenetration, VisualHullPenetration, reference_depths, reference_visual_depths,
            )
            penetration_measure = self.options.get("penetration_measure", "collision")
            if penetration_measure == "visual_hull":
                penetration_monitor = VisualHullPenetration(
                    env.scene, samples_per_shape=int(self.options.get("penetration_samples_per_shape", 8)),
                )
                reference_frames, reference_overlap = reference_visual_depths(
                    env, penetration_monitor,
                    stride=int(self.options.get("penetration_reference_stride", 4)),
                )
            elif penetration_measure == "collision":
                penetration_monitor = HandObjectPenetration(env.scene)
                contact_active = np.stack([
                    np.asarray(self.eval_setup.reference.contact_active(side)).reshape(binding.num_frames, -1).any(axis=1)
                    for side in penetration_monitor.sides
                ], axis=1)
                reference_frames, reference_overlap = reference_depths(
                    env, penetration_monitor, contact_active,
                    stride=int(self.options.get("penetration_reference_stride", 4)),
                )
            else:
                raise ValueError(f"unknown penetration_measure: {penetration_measure}")
            rollout_stride = int(self.options.get("penetration_rollout_stride", 4))
            if rollout_stride < 1:
                raise ValueError("penetration_rollout_stride must be positive")
            rollout_overlap = []
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        from flash_chord.runtime.jax import to_jax
        initial_observation = env.reset(frame_id=0)
        if initial_q_delta is not None:
            initial_q = env.state_0.joint_q.numpy().reshape(env.world_count,-1)
            initial_q[:,:binding.num_joint_q] += np.asarray(initial_q_delta)[None,:]
            initial_q[:,self.projector.qids] = np.clip(initial_q[:,self.projector.qids],self.projector.lower,self.projector.upper)
            for state in (env.state_0,env.state_1):
                state.joint_q.assign(initial_q.reshape(-1))
                newton.eval_fk(env.model,state.joint_q,state.joint_qd,state)
            initial_observation = env.observation_strategy.compute(env.state_0)
        observation = jnp.array(to_jax(initial_observation, (env.world_count, env.observation_dim)), copy=True)
        if voc_probe_scale is not None:
            np.testing.assert_allclose(env.object_control_scale.numpy(),voc_probe_scale,atol=1e-7)
            if env.command.voc_body_ids_w.size != env.world_count * env.command.layout.num_objects:
                raise AssertionError("VOC root-body count does not match the object roster")
        state = ActorInferenceState(learner.actor.params, learner.actor.batch_stats)
        key = jax.random.PRNGKey(seed)
        if self.actor is None:
            self.actor, _ = create_actor_template(self.training, observation.shape[1], adapter.action_dim)
            self.policy = compile_policy_action(self.actor, state, observation, key,
                deterministic=bool(self.options.get("study_variant")),
                action_transform=self.training.network.action_transform, action_scale=self.training.network.action_scale)
        positions, quats, indices, obj_errors, angles, wrench_errors = [], [], [], [], [], []
        upper_wrench = self.new_wrench_accumulator(env) if hasattr(self, 'new_wrench_accumulator') else None
        if self.contact_proximity is not None:
            proximity_frame_sums, proximity_frame_counts = [], []
            proximity_sums = np.zeros((env.world_count, len(self.contact_proximity.sides),
                                       self.contact_proximity.num_objects))
            proximity_counts = np.zeros_like(proximity_sums, dtype=np.int64)
            contact_chunk_feedback = return_feedback and self.options.get("study_variant") == "chunk"
            if contact_chunk_feedback:
                from flash_chord.retargeting.chunk_feedback import chunk_labels
                _, contact_chunk_count = chunk_labels(
                    [0], binding.num_frames, binding.fps,
                    float(self.options.get("chunk_seconds", .5))
                )
                contact_vectors = np.zeros((contact_chunk_count, *proximity_sums.shape, 3))
                contact_counts = np.zeros((contact_chunk_count, *proximity_counts.shape), dtype=np.int64)
        study = bool(self.options.get("study_variant"))
        object_displacements, position_frames, orientation_frames, wrench_frames, wrench_active_frames = [], [], [], [], []
        for step_index in range(binding.num_frames - 1):
            frame = env.timestep.numpy().copy()
            action, key = self.policy(state, observation, key)
            transition = adapter.step(action)
            self.validation_steps += env.world_count
            observation = jnp.array(transition.observation, copy=True)
            if getattr(self, "penetration_aware", False) and step_index % rollout_stride == 0:
                if penetration_measure == "visual_hull":
                    rollout_overlap.append(penetration_monitor.frame_depths(env.state_0))
                else:
                    active_frames = np.clip(np.asarray(env.timestep.numpy()).reshape(-1), 0, len(contact_active) - 1)
                    rollout_overlap.append(penetration_monitor.frame_depths(
                        env.state_0, penetration_monitor.collide(env.state_0), contact_active[active_frames],
                    ))
            bodies = env.state_0.body_q.numpy().reshape(env.world_count, -1, 7)
            p, q = _frame_poses(bodies, self.frames)
            positions.append(p); quats.append(q); indices.append(frame)
            ids = env.command.body_ids_w.numpy().reshape(-1)
            actual = env.state_0.body_q.numpy()[ids]
            if self.contact_proximity is not None:
                hand_count = len(self.contact_proximity.sides)
                if p.shape[1] != hand_count * 6:
                    raise ValueError("contact proximity expects wrist plus five fingertips per hand")
                proximity_result = self.contact_proximity.frame(
                    frame, p.reshape(env.world_count, hand_count, 6, 3)[:, :, 1:],
                    actual.reshape(env.world_count, self.contact_proximity.num_objects, 7),
                    return_vectors=contact_chunk_feedback,
                )
                frame_sums, frame_counts = proximity_result[:2]
                proximity_sums += frame_sums
                proximity_counts += frame_counts
                if return_feedback:
                    proximity_frame_sums.append(frame_sums.copy())
                    proximity_frame_counts.append(frame_counts.copy())
                if contact_chunk_feedback:
                    chunk_ids, _ = chunk_labels(
                        frame, binding.num_frames, binding.fps,
                        float(self.options.get("chunk_seconds", .5))
                    )
                    for world, chunk in enumerate(chunk_ids):
                        contact_vectors[chunk, world] += proximity_result[2][world]
                        contact_counts[chunk, world] += frame_counts[world]
            target_p = env.command.body_target_pos_w.numpy().reshape(-1,3)
            target_q = env.command.body_target_quat_w.numpy().reshape(-1,4)
            frame_pos_error = np.linalg.norm(actual[:,:3]-target_p, axis=-1)
            obj_errors.extend(frame_pos_error.tolist())
            # Command quaternions are Warp xyzw, as are Newton body transforms.
            frame_angle = (Rotation.from_quat(actual[:,3:]) * Rotation.from_quat(target_q).inv()).magnitude()
            angles.extend(frame_angle.tolist())
            if study or return_feedback:
                object_displacements.append((actual[:,:3]-target_p).reshape(env.world_count,-1,3).mean(axis=1))
                position_frames.append(frame_pos_error.reshape(env.world_count,-1).mean(axis=1))
                orientation_frames.append(frame_angle.reshape(env.world_count,-1).mean(axis=1))
            current = env.objective.current_wrench_support.numpy().reshape(env.world_count,-1)
            target = env.objective.wrench_support_target.numpy().reshape(env.world_count,-1)
            if upper_wrench is not None:
                upper_wrench.add(current, target)
            active = np.mean(target**2, axis=1) > 1e-8
            if np.any(active):
                wrench_errors.extend((np.mean((current[active]-target[active])**2,axis=1) /
                    np.maximum(np.mean(target[active]**2,axis=1), 1e-6)).tolist())
            if study or return_feedback:
                frame_wrench = np.zeros(env.world_count, dtype=np.float64)
                if np.any(active):
                    frame_wrench[active] = np.mean((current[active]-target[active])**2,axis=1) / np.maximum(
                        np.mean(target[active]**2,axis=1), 1e-6)
                wrench_frames.append(frame_wrench)
                wrench_active_frames.append(active.copy())
        idx = np.asarray(indices).reshape(-1)
        achieved = body_motion(np.concatenate(positions), np.concatenate(quats))
        commanded = binding_motion(binding)
        pe = np.asarray(commanded.position)[idx] - np.asarray(achieved.position)
        rr = np.asarray(commanded.rotation)[idx] @ np.swapaxes(np.asarray(achieved.rotation),-1,-2)
        re = Rotation.from_matrix(rr.reshape(-1,3,3)).magnitude()
        metrics = dict(robot_position_mse=float(np.mean(pe**2)), robot_rotation_mse=float(np.mean(re**2)),
            wrench_mse=float(np.mean(wrench_errors)) if wrench_errors else 0.0,
            object_position_mean_m=float(np.mean(obj_errors)), object_position_p95_m=float(np.percentile(obj_errors,95)),
            object_orientation_mean_rad=float(np.mean(angles)),
            object_success_fraction=float(np.mean((np.asarray(obj_errors)<.1)&(np.asarray(angles)<.5))))
        if upper_wrench is not None:
            if env.world_count != 1:
                raise ValueError('response confirmation requires one world')
            metrics.update(upper_wrench.summaries()[0])
        if self.contact_proximity is not None:
            from flash_chord.retargeting.contact_proximity import summarize_contact_proximity
            metrics.update(summarize_contact_proximity(proximity_sums, proximity_counts))
        object_count = env.command.body_ids_w.numpy().size // env.world_count
        per_pos = np.asarray(obj_errors).reshape(-1, env.world_count, object_count)
        per_angle = np.asarray(angles).reshape(-1, env.world_count, object_count)
        if hasattr(self, 'trajectory_metrics'):
            metrics.update(self.trajectory_metrics(per_pos.mean(axis=1), per_angle.mean(axis=1)))
        metrics["per_object_position_p95_m"] = np.percentile(per_pos,95,axis=(0,1)).tolist()
        metrics["per_object_orientation_p95_rad"] = np.percentile(per_angle,95,axis=(0,1)).tolist()
        metrics["per_object_gate_success_fraction"] = ((per_pos<=.05)&(per_angle<=np.deg2rad(20))).mean(axis=(0,1)).tolist()
        metrics["robot_position_rms_m"] = float(np.sqrt(np.mean(np.sum(pe**2,axis=-1))))
        metrics["all_errors_finite"] = bool(np.isfinite(per_pos).all() and np.isfinite(per_angle).all() and np.isfinite(pe).all())
        if voc_probe_scale is not None:
            np.testing.assert_allclose(env.object_control_scale.numpy(),voc_probe_scale,atol=1e-7)
            metrics["probe_applied_voc_scale"] = float(env.object_control_scale.numpy()[0])
            metrics["voc_object_count"] = int(env.command.layout.num_objects)
        metrics["full_horizon"] = bool(len(indices)==binding.num_frames-1 and all(np.all(frame==i) for i,frame in enumerate(indices)))
        metrics["per_object_position_mean_m"] = np.nan_to_num(per_pos, nan=1., posinf=1.).mean(axis=(0,1)).tolist()
        metrics["per_object_orientation_mean_rad"] = np.nan_to_num(per_angle, nan=np.pi, posinf=np.pi).mean(axis=(0,1)).tolist()
        if study:
            from flash_chord.retargeting.ttsa_editor import robust_world_summary
            metrics.update(robust_world_summary(np.stack(position_frames), np.stack(orientation_frames),
                                                np.stack(wrench_frames),np.stack(wrench_active_frames)))
            metrics["robot_position_robust_mse"] = float(np.mean(np.minimum(pe**2, .1**2)))
        if getattr(self, "penetration_aware", False):
            from flash_chord.retargeting.penetration import penetration_summary
            metrics.update({f"reference_{key}": value for key, value in penetration_summary(reference_overlap).items()})
            metrics.update({f"rollout_{key}": value for key, value in penetration_summary(np.stack(rollout_overlap)).items()})
        if return_feedback:
            from flash_chord.retargeting.ttsa_editor import signed_feedback_summary, consensus_frame_scores
            # The upper editor changes one reusable hand trajectory. Measure its
            # signed error relative to the fixed object target in each world.
            hand_error = pe.reshape(len(positions), env.world_count, *pe.shape[1:])
            relative_error = hand_error + np.stack(object_displacements)[:, :, None, :]
            feedback = dict(signed_bins=signed_feedback_summary(relative_error),
                            frame_per_object_position_m=per_pos,
                            frame_per_object_orientation_rad=per_angle,
                            signed_frames=relative_error.mean(axis=2),
                            frame_object_position_m=np.stack(position_frames),
                            frame_object_orientation_rad=np.stack(orientation_frames),
                            frame_wrench_mse=np.stack(wrench_frames),
                            frame_wrench_active=np.stack(wrench_active_frames),
                            # Large mean with little world disagreement scores
                            # above large-but-opposing per-world corrections.
                            frame_scores=consensus_frame_scores(relative_error))
            if self.contact_proximity is not None:
                feedback["frame_contact_sums_m"] = np.stack(proximity_frame_sums)
                feedback["frame_contact_counts"] = np.stack(proximity_frame_counts)
            if self.options.get("study_variant") == "chunk":
                from flash_chord.retargeting.chunk_feedback import summarize_chunks
                feedback["signed_bins"], feedback["chunk_sample_counts"] = summarize_chunks(
                    relative_error, np.asarray(indices), binding.num_frames, binding.fps,
                    float(self.options.get("chunk_seconds", .5)))
                if self.contact_proximity is not None:
                    summed = contact_vectors.sum(axis=1)
                    counts = contact_counts.sum(axis=1)
                    feedback["contact_direction_bins"] = np.divide(
                        summed, np.maximum(counts[..., None], 1)
                    ).astype(np.float32)
                    feedback["contact_direction_counts"] = counts.sum(axis=(1, 2))
            if getattr(self, "penetration_aware", False):
                overlap = np.stack(rollout_overlap)
                feedback["frame_penetration_m"] = overlap
                feedback["penetration_frame_ids"] = np.arange(1, binding.num_frames, rollout_stride)
                feedback["penetration_bins"] = np.stack([
                    np.stack((overlap[ids].mean(axis=(0, 1)),
                              overlap[ids].mean(axis=0).std(axis=0)))
                    for ids in np.array_split(np.arange(len(overlap)), 16)
                ])
                feedback["penetration_frame_scores"] = np.interp(
                    np.arange(binding.num_frames), reference_frames,
                    reference_overlap.mean(axis=(1, 2)),
                ) + .5 * np.interp(
                    np.arange(binding.num_frames),
                    np.arange(1, binding.num_frames, rollout_stride),
                    overlap.mean(axis=(1, 2)),
                )
            return metrics, idx, achieved, feedback
        return metrics, idx, achieved

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
        jax.block_until_ready(learner)
        started = time.perf_counter()
        evaluation_start = self.validation_steps
        seed = self.training.seed + step
        baseline, indices, achieved = self.rollout(self.binding, learner, seed)
        record = dict(step=step, mode=self.mode, version_before=self.version, baseline=baseline, candidates=[])
        record["feedback_phase_counts"] = {str(p):int(np.sum(self.phases[indices]==p)) for p in np.unique(self.phases)}
        weight = float(self.options.get("wrench_weight", 1.0)) if self.mode == "reactor_wrench" else 0.0
        accepted = False
        if self.mode != "fixed":
            count = min(256, min(record["feedback_phase_counts"].values()))
            if count < 1:
                raise ValueError("insufficient feedback coverage for a reference update")
            selected = np.concatenate([self.rng.choice(np.flatnonzero(self.phases[indices]==p), count, replace=False)
                                       for p in np.unique(self.phases)])
            source_batch = jax.tree.map(lambda x: x[indices[selected]], self.source)
            achieved_batch = jax.tree.map(lambda x: x[selected], achieved)
            proposal, loss = outer_step(self.theta, source_batch, achieved_batch, self.nominal,
                jnp.zeros(len(selected),dtype=jnp.int32), self.heights, jnp.ones(len(selected),dtype=bool),
                learning_rate=float(self.options.get("learning_rate", .001)),
                limits=ReactorLimits(position=.02,orientation=.1,height=.01))
            # Both adaptive arms see identical proposals. Only wrench scoring differs.
            proposals = [proposal]
            direction = self.rng.normal(size=proposal.position.shape).astype(np.float32)
            direction /= np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-8)
            for sign in (-1,1):
                proposals.append(replace(proposal,position=jnp.clip(proposal.position+sign*.002*direction,-.02,.02)))
            best = candidate_score(baseline, weight)
            best_pair = None
            for proposal_index, theta in enumerate(proposals):
                print(f"[bilevel] step={step} projecting candidate {proposal_index+1}/{len(proposals)}", flush=True)
                targets = map_motion(self.source,self.nominal,theta,self.clip_ids,self.heights)
                try:
                    candidate = self.projector.project(targets)
                    metrics, _, _ = self.rollout(candidate, learner, seed)
                    score = candidate_score(metrics, weight)
                    passed = passes_task_gate(metrics, baseline)
                    print(f"[bilevel] candidate={proposal_index+1} score={score:.6f} task_gate={passed}", flush=True)
                    record["candidates"].append(dict(metrics=metrics,score=score,task_gate=passed))
                    if passed and score < best - 1e-6:
                        best, best_pair = score, (theta,candidate)
                except ValueError as error:
                    record["candidates"].append(dict(rejected=str(error)))
            record["outer_tracking_loss"] = float(loss)
            if best_pair is not None:
                # Independent action-noise seed prevents accepting selection-only gains.
                control, _, _ = self.rollout(self.binding, learner, seed+100000)
                check, _, _ = self.rollout(best_pair[1], learner, seed+100000)
                record["validation"] = dict(baseline=control,candidate=check)
                if passes_task_gate(check,control) and candidate_score(check,weight) < candidate_score(control,weight):
                    self.theta,self.binding = best_pair
                    self.version += 1
                    self.publish(step)
                    accepted = True
        record.update(accepted=accepted,version_after=self.version,seconds=time.perf_counter()-started,
                      upper_simulation_transitions=self.validation_steps-evaluation_start,
                      cumulative_upper_simulation_transitions=self.validation_steps)
        if command_digest(self.setup.reference) != self.command_sha256:
            raise RuntimeError("upper update mutated object/contact commands")
        with (self.output.parent/"upper_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        print(f"[bilevel] mode={self.mode} step={step} accepted={accepted} version={self.version} seconds={record['seconds']:.1f}",flush=True)
        if not accepted:
            return env  # runner still resets observations and clears replay in all arms
        # Rebuild ALL consumers, not only objective targets. Fixed control uses the
        # same reset/replay-clear schedule to isolate the reference-update effect.
        replacement = RLEnv(replace(self.setup.scene,robot_reference=self.binding), self.setup.reference,
                            config=deepcopy(env.env.config))
        assert_device_binding(replacement,self.binding)
        np.testing.assert_array_equal(replacement.command.reference.body_pos_w.numpy(),
                                      env.env.command.reference.body_pos_w.numpy())
        np.testing.assert_array_equal(replacement.command.reference.body_quat_w.numpy(),
                                      env.env.command.reference.body_quat_w.numpy())
        np.testing.assert_array_equal(replacement.objective.ref_wrench_support.numpy(),
                                      env.env.objective.ref_wrench_support.numpy())
        replacement.reset(); replacement.capture_step(); replacement.reset()
        return WarpRLEnv(replacement,publish_diagnostics=False,publish_step_metrics=False)
