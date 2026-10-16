"""Parallel unchanged-reference probes used only to locate failure windows."""
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np

from flash_chord.retargeting.failure_consensus import perturb_initial_joints


class BaselineFailureProbe:
    def __init__(self, editor, *, embodiment, collision, worlds=16):
        from flash_chord.scene.setup import setup_scene
        from flash_chord.retargeting.reference_snapshot import command_digest
        if not 2 <= worlds <= 64:
            raise ValueError('baseline probe worlds must be between 2 and 64')
        self.worlds = worlds
        scene, task = editor.config['scene'], editor.config['task']
        self.setup = setup_scene(parquet=task['parquet'], control_fps=editor.eval_config.sim.fps,
            motion_speed=task['motion_speed'], embodiment=embodiment, collision=collision,
            world_count=worlds, include_support=scene['support'],
            decompose_objects=scene['decompose_objects'], contact_friction=scene.get('contact_friction', 1.),
            object_free_joint_damping=scene.get('object_free_joint_damping', 0.),
            object_initial_pose_overrides=scene.get('object_initial_pose_overrides'),
            source_frame_playback=task['source_frame_playback'],
            motion_start_frame=task['motion_start_frame'], motion_end_frame=task['motion_end_frame'])
        if command_digest(self.setup.reference) != editor.command_sha256:
            raise ValueError('baseline probe changed object commands')
        self.policy = None

    def __call__(self, editor, learner, seed):
        import jax
        import jax.numpy as jnp
        import newton
        from scipy.spatial.transform import Rotation
        from flash_chord.envs.rl import RLEnv
        from flash_chord.training.environment import WarpRLEnv
        from flash_chord.runtime.jax import to_jax
        from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
        from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action

        started = time.perf_counter()
        binding = editor.binding
        env = RLEnv(replace(self.setup.scene, robot_reference=binding), self.setup.reference,
                    config=deepcopy(editor.eval_config))
        env.reset(frame_id=0)
        if env.device.is_cuda:
            env.capture_step()
        env.reset(frame_id=0)
        if env.config.auto_reset:
            raise ValueError('baseline probe must evaluate the complete episode without auto reset')
        np.testing.assert_allclose(env.object_control_scale.numpy(), 0., atol=1e-7)
        initial = perturb_initial_joints(binding.joint_q[0], editor.projector.layout.scalar_joints.names,
            editor.projector.qids, editor.projector.lower, editor.projector.upper, self.worlds, seed)
        q = env.state_0.joint_q.numpy().reshape(self.worlds, -1)
        q[:, :binding.num_joint_q] = initial
        for state in (env.state_0, env.state_1):
            state.joint_q.assign(q.reshape(-1))
            newton.eval_fk(env.model, state.joint_q, state.joint_qd, state)
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        obs = jnp.array(to_jax(env.observation_strategy.compute(env.state_0),
                              (self.worlds, env.observation_dim)), copy=True)
        state = ActorInferenceState(learner.actor.params, learner.actor.batch_stats)
        key = jax.random.PRNGKey(seed)
        if self.policy is None:
            actor, _ = create_actor_template(editor.training, obs.shape[1], adapter.action_dim)
            self.policy = compile_policy_action(actor, state, obs, key, deterministic=True,
                action_transform=editor.training.network.action_transform,
                action_scale=editor.training.network.action_scale)
        positions, orientations = [], []
        objects = env.command.layout.num_objects
        ids = env.command.body_ids_w.numpy().reshape(-1)
        for frame in range(binding.num_frames-1):
            np.testing.assert_array_equal(env.timestep.numpy(), np.full(self.worlds, frame))
            action, key = self.policy(state, obs, key)
            transition = adapter.step(action)
            obs = jnp.array(transition.observation, copy=True)
            actual = env.state_0.body_q.numpy()[ids].reshape(self.worlds, objects, 7)
            tp = env.command.body_target_pos_w.numpy().reshape(self.worlds, objects, 3)
            tq = env.command.body_target_quat_w.numpy().reshape(-1, 4)
            positions.append(np.linalg.norm(actual[:, :, :3]-tp, axis=-1))
            orientations.append((Rotation.from_quat(actual[:, :, 3:].reshape(-1, 4)) *
                                 Rotation.from_quat(tq).inv()).magnitude().reshape(self.worlds, objects))
        p, a = np.asarray(positions), np.asarray(orientations)
        if not np.isfinite(p).all() or not np.isfinite(a).all():
            raise FloatingPointError('nonfinite baseline ensemble trajectory')
        editor.validation_steps += self.worlds*(binding.num_frames-1)
        metadata = dict(baseline_probe_seed=int(seed), baseline_probe_seconds=time.perf_counter()-started,
                        baseline_probe_worlds=self.worlds, baseline_probe_parallel=True,
                        baseline_probe_voc_scale=0., baseline_probe_reference_version=editor.version,
                        baseline_probe_controller='unchanged current lower actor',
                        baseline_probe_translation_m=.001, baseline_probe_finger_degrees=.5,
                        baseline_probe_distinct_resets=int(len(np.unique(initial, axis=0))))
        if metadata['baseline_probe_distinct_resets'] != self.worlds:
            raise ValueError('baseline probe resets are not distinct')
        return p, a, initial-binding.joint_q[0], metadata
