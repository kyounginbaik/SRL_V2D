"""Practice as bilevel optimization, for a floating hand writing with a pen.

Lower level: FlashSAC trains a policy to track the current hand reference and the pen goal.
Upper level: every `interval` steps, freeze the policy and try edits to the hand reference.
    Pen goals are never edited, so the task itself cannot change.
Signal: the writing itself, judged at the assistance level the policy is currently training at.
    Pass fraction first, then the share of frames still on track, then tip error and time on the board.
Proposals come from what practice just showed:
    copy: move the reference toward what the policy's hand actually did on its best world.
    grip: shift the whole grip on the pen and close or open the fingers.
    local: correct the wrist around the frame where the pen went wrong.
A winner must beat the current reference again on a fresh seed before it is published.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.retargeting.manipulation_bilevel import LocalProjector
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
from flash_chord.training.flash_sac.evaluation import compile_policy_action, create_actor_template

# The pass rule fixed before any method ran (write/wmetric.py).
TIP_LIMIT, ON_BOARD_MIN, BOARD_BAND = 0.01, 0.8, 0.005
# When a world counts as having failed: the training termination limits, plus losing the writing.
LOST_P, LOST_R, LOST_TIP = 0.10, 0.5, 0.03
# How far the reference may drift from where it started.
MAX_P, MAX_R, MAX_FINGER = 0.05, np.radians(40.0), 0.4


class PracticeEditor:
    def __init__(self, *, options, setup, embodiment, collision, training, environment_config, checkpoint_metadata, output_dir):
        self.options = dict(options)
        if training.resume:
            raise ValueError("resume is not supported for the practice loop")
        self.interval = int(self.options["interval"])
        self.next_update = int(self.options.get("warmup", self.interval))
        self.count = int(self.options.get("candidates", 10))
        self.training, self.setup, self.metadata = training, setup, checkpoint_metadata
        self.config = deepcopy(environment_config)
        self.output = Path(output_dir) / "references"
        self.output.mkdir(parents=True, exist_ok=True)
        self.baseline = self.binding = setup.scene.robot_reference
        self.command = command_digest(setup.reference)
        self.projector = LocalProjector(embodiment, self.baseline)
        self.hand = next(h for h in self.projector.layout.hands if h.side == "right")
        self.rng = np.random.default_rng(training.seed + 10000)
        self.kind = str(self.options.get("task_kind", "write"))
        if self.kind == "write":
            board = json.loads(Path(self.options["board"]).read_text())
            self.normal, self.surface, self.tip_local = np.array(board["normal"]), np.array(board["surface_point"]), np.array(board["tip_local"])
        self.art_offset = None
        self.train_voc = instantiate_typed(OmegaConf.create(self.config["env"]), RLEnvConfig).voc
        evaluation = deepcopy(self.config)
        apply_evaluation_composition(evaluation, world_count=int(self.options.get("eval_worlds", 16)))
        self.eval_config = instantiate_typed(OmegaConf.create(evaluation["env"]), RLEnvConfig)
        scene, task = evaluation["scene"], evaluation["task"]
        self.eval_setup = setup_scene(
            parquet=task["parquet"], control_fps=self.eval_config.sim.fps, motion_speed=task["motion_speed"],
            embodiment=embodiment, collision=collision, world_count=scene["world_count"],
            include_support=scene["support"], decompose_objects=scene["decompose_objects"],
            contact_friction=scene.get("contact_friction", 1.0),
            object_free_joint_damping=scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=task["source_frame_playback"],
            motion_start_frame=task["motion_start_frame"], motion_end_frame=task["motion_end_frame"],
        )
        lower = self.projector.model.joint_limit_lower.numpy()
        upper = self.projector.model.joint_limit_upper.numpy()
        self.finger_limits = np.stack([lower[list(self.hand.finger_dof_ids)], upper[list(self.hand.finger_dof_ids)]], axis=1)
        self.actor = self.policy = None
        self.version = 0
        self.upper_transitions = 0
        self.goal = None
        self._publish(0)

    def _tip(self, position, quat_xyzw):
        return R.from_quat(quat_xyzw).apply(self.tip_local) + position

    def _prepare_goal(self, position, quat_xyzw):
        """Writing frames come from the goal itself: tip on the fitted plane, between first and last touch."""
        self.goal = (position.copy(), R.from_quat(quat_xyzw))
        tip = self._tip(position, quat_xyzw)
        height = (tip - self.surface) @ self.normal
        low = np.percentile(height, 20)
        on = height < low + 0.002
        first, last = np.flatnonzero(on)[[0, -1]]
        self.on = on
        self.on[:first] = False
        self.on[last + 1:] = False
        self.height0 = float(np.median(height[self.on]))
        self.goal_tip = tip
        print(f"[practice] goal frames={len(tip)} writing frames={int(self.on.sum())} touch={first}-{last} height0={self.height0 * 100:.3f}cm", flush=True)

    def _prepare_scissors(self, fps, frames):
        """Opening-angle goal and its cut events, from the reference articulation at the control rate."""
        from scipy.signal import find_peaks
        th = np.asarray(self.eval_setup.reference.object_articulation(), dtype=float)[:, 0]
        if len(th) != frames + 1:
            th = np.interp(np.linspace(0, 1, frames + 1), np.linspace(0, 1, len(th)), th)
        self.goal_angle = np.degrees(th[1:])
        dist = max(1, int(0.6 * fps))
        self.closes, _ = find_peaks(-self.goal_angle, prominence=10, distance=dist)
        self.opens, _ = find_peaks(self.goal_angle, prominence=10, distance=dist)
        if self.goal_angle[0] > 30 and self.goal_angle[0] > self.goal_angle[1] and 0 not in self.opens:
            self.opens = np.r_[0, self.opens]
        self.window = int(round(0.4 * fps))
        print(f"[practice] scissors goal frames={frames} opens={self.opens.tolist()} closes={self.closes.tolist()}", flush=True)

    def measure_scissors(self, poses, angles):
        """Scissors rule (scissors/smetric.py) without the overlap check. tip_cm holds the median angle error in degrees, on_board is unused."""
        frames, worlds = poses.shape[:2]
        gp, gr = self.goal
        gp, gr = gp[:frames], gr[:frames]
        ga = self.goal_angle[:frames]
        err_a, frontier, passed, kept = [], [], [], []
        for w in range(worlds):
            x = poses[:, w]
            quat = np.array(x[:, 3:7], dtype=float)
            bad = ~np.isfinite(x).all(axis=1) | (np.linalg.norm(np.nan_to_num(quat), axis=1) < 1e-6) | ~np.isfinite(angles[:, w])
            quat[bad] = (0.0, 0.0, 0.0, 1.0)
            pos = np.nan_to_num(x[:, :3], nan=1e3, posinf=1e3, neginf=-1e3)
            rot = R.from_quat(quat)
            ep = np.linalg.norm(pos - gp, axis=1)
            er = (rot * gr.inv()).magnitude()
            a = np.degrees(np.nan_to_num(angles[:, w], nan=1e3))
            ea = np.abs(a - ga)
            lost = bad | (ep > LOST_P) | (er > LOST_R) | (ea > 15.0)
            f = int(np.flatnonzero(lost)[0]) if lost.any() else frames
            W_ = self.window
            ok = [a[max(0, i - W_):i + W_ + 1].max() >= ga[i] - 8 for i in self.opens if i < frames] + [a[max(0, i - W_):i + W_ + 1].min() <= ga[i] + 8 for i in self.closes if i < frames]
            frontier.append(f)
            kept.append(float((~lost).mean()))
            err_a.append(float(np.median(ea)))
            passed.append(bool(np.median(ep) < 0.02 and np.degrees(np.median(er)) < 15 and np.median(ea) < 6 and all(ok)
                               and len(ok) == len(self.opens) + len(self.closes) and ep[-1] < 0.05 and not bad.any()))
        return dict(
            passed=float(np.mean(passed)), kept=float(np.mean(kept)), frontier=int(np.median(frontier)), frontier_max=int(np.max(frontier)),
            tip_cm=float(np.median(err_a)), on_board=1.0,
            best_world=int(np.lexsort((err_a, -np.asarray(frontier)))[0]),
        )

    def measure(self, poses):
        """Per-world writing score at the control rate. poses: frames, worlds, xyz + xyzw."""
        frames, worlds = poses.shape[:2]
        gp, gr = self.goal
        gp, gr = gp[:frames], gr[:frames]
        tip_goal, on = self.goal_tip[:frames], self.on[:frames]
        tip_med, on_board, frontier, passed, kept = [], [], [], [], []
        for w in range(worlds):
            x = poses[:, w]
            quat = np.array(x[:, 3:7], dtype=float)
            bad = ~np.isfinite(x).all(axis=1) | (np.linalg.norm(np.nan_to_num(quat), axis=1) < 1e-6)
            quat[bad] = (0.0, 0.0, 0.0, 1.0)
            pos = np.nan_to_num(x[:, :3], nan=1e3, posinf=1e3, neginf=-1e3)
            rot = R.from_quat(quat)
            ep = np.linalg.norm(pos - gp, axis=1)
            er = (rot * gr.inv()).magnitude()
            tip = rot.apply(self.tip_local) + pos
            te = np.linalg.norm(tip - tip_goal, axis=1)
            ob = np.abs((tip - self.surface) @ self.normal - self.height0) < BOARD_BAND
            lost = bad | (ep > LOST_P) | (er > LOST_R) | (on & (te > LOST_TIP))
            f = int(np.flatnonzero(lost)[0]) if lost.any() else frames
            tm, of = float(np.median(te[on])), float(ob[on].mean())
            frontier.append(f)
            kept.append(float((~lost).mean()))
            tip_med.append(tm)
            on_board.append(of)
            passed.append(tm < TIP_LIMIT and of >= ON_BOARD_MIN and ep[-1] < LOST_P and not bad.any())
        return dict(
            passed=float(np.mean(passed)), kept=float(np.mean(kept)), frontier=int(np.median(frontier)), frontier_max=int(np.max(frontier)),
            tip_cm=float(np.median(tip_med) * 100), on_board=float(np.median(on_board)),
            best_world=int(np.lexsort((tip_med, -np.asarray(frontier)))[0]),
        )

    @staticmethod
    def score(m):
        return (m["passed"], m["kept"], m["on_board"] - m["tip_cm"])

    @staticmethod
    def better(a, b):
        """A real gain only: publishing clears the replay buffer, so small differences are not worth it."""
        if a["passed"] != b["passed"]:
            return a["passed"] >= b["passed"] + 0.1
        if abs(a["kept"] - b["kept"]) >= 0.03:
            return a["kept"] > b["kept"]
        return a["tip_cm"] <= 0.95 * b["tip_cm"] and a["on_board"] >= b["on_board"] - 0.02

    def rollout(self, binding, learner, seed, voc):
        """Frozen policy, whole clip from the first frame, at this assistance level."""
        eval_config = deepcopy(self.eval_config)
        if voc > 0.0:
            eval_config.voc = deepcopy(self.train_voc)
            eval_config.voc_scale = float(voc)
        env = RLEnv(replace(self.eval_setup.scene, robot_reference=binding), self.eval_setup.reference, config=eval_config)
        env.reset(frame_id=0)
        if env.device.is_cuda:
            env.capture_step()
        env.reset(frame_id=0)
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        from flash_chord.runtime.jax import to_jax

        obs = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
        state = ActorInferenceState(learner.actor.params, learner.actor.batch_stats)
        key = jax.random.PRNGKey(seed)
        if self.actor is None:
            self.actor, _ = create_actor_template(self.training, obs.shape[1], adapter.action_dim)
            self.policy = compile_policy_action(
                self.actor, state, obs, key, deterministic=True,
                action_transform=self.training.network.action_transform, action_scale=self.training.network.action_scale,
            )
        W = env.world_count
        ids = env.command.body_ids_w.numpy().reshape(-1)
        poses, goals, joints, angles = [], [], [], []
        if self.kind == "scissors" and self.art_offset is None:
            labels = [str(x) for x in env.model.joint_label]
            j = next(i for i, x in enumerate(labels) if x.split("/")[-1] == "rotation")
            self.art_offset = int(env.model.joint_q_start.numpy()[j]) % (env.model.joint_coord_count // W)
        for _ in range(binding.num_frames - 1):
            action, key = self.policy(state, obs, key)
            obs = jnp.array(adapter.step(action).observation, copy=True)
            jax.block_until_ready(obs)
            poses.append(env.state_0.body_q.numpy()[ids].reshape(W, -1, 7)[:, 0].copy())
            goals.append(np.concatenate([env.command.body_target_pos_w.numpy().reshape(W, -1, 3)[0, 0], env.command.body_target_quat_w.numpy().reshape(W, -1, 4)[0, 0]]))
            jq = env.state_0.joint_q.numpy().reshape(W, -1)
            joints.append(jq[:, : binding.num_joint_q].copy())
            if self.kind == "scissors":
                angles.append(jq[:, self.art_offset].copy())
        self.upper_transitions += (binding.num_frames - 1) * W
        poses, goals, joints = np.asarray(poses), np.asarray(goals), np.asarray(joints)
        if self.kind == "scissors":
            if self.goal is None:
                self.goal = (goals[:, :3].copy(), R.from_quat(goals[:, 3:]))
                self._prepare_scissors(eval_config.sim.fps, len(goals))
            return self.measure_scissors(poses, np.asarray(angles)), poses, joints
        if self.goal is None:
            self._prepare_goal(goals[:, :3], goals[:, 3:])
        return self.measure(poses), poses, joints

    def _wrist(self, q):
        hand = self.hand
        return q[:, hand.wrist_pos_q_ids].astype(float), R.from_quat(q[:, hand.wrist_orient_q_id : hand.wrist_orient_q_id + 4].astype(float))

    def _with_wrist(self, q, position, rotation):
        hand = self.hand
        q = q.copy()
        q[:, hand.wrist_pos_q_ids] = position
        q[:, hand.wrist_orient_q_id : hand.wrist_orient_q_id + 4] = rotation.as_quat()
        return q

    def bind(self, q):
        hand, base = self.hand, self.baseline
        q = q.copy()
        q[:, hand.finger_q_ids] = np.clip(q[:, hand.finger_q_ids], self.finger_limits[:, 0], self.finger_limits[:, 1])
        target = base.joint_target.copy()
        position, rotation = self._wrist(q)
        target[:, hand.wrist_pos_dof_ids] = position
        target[:, hand.wrist_orient_dof_ids] = rotation.as_rotvec()
        target[:, hand.finger_dof_ids] = q[:, hand.finger_q_ids]
        return build_robot_reference_binding(self.projector.builder, self.projector.layout, q.astype(np.float32), target.astype(np.float32), base.fps)

    def in_bounds(self, q):
        """A refinement stays a refinement of the reference it started from."""
        hand, base = self.hand, self.baseline.joint_q
        p0, r0 = self._wrist(base)
        p1, r1 = self._wrist(q)
        return (np.linalg.norm(p1 - p0, axis=1).max() <= MAX_P and (r1 * r0.inv()).magnitude().max() <= MAX_R
                and np.abs(q[:, hand.finger_q_ids] - base[:, hand.finger_q_ids]).max() <= MAX_FINGER)

    def propose(self, measured, poses, joints):
        hand, rng, q = self.hand, self.rng, np.asarray(self.binding.joint_q, dtype=float)
        n = len(q)
        frames = poses.shape[0]
        frontier = min(measured["frontier"], frames - 1)
        world = measured["best_world"]
        out = []
        # Copy: pull the reference toward what the hand did, in the goal's frame, up to where the pen was still on track.
        gp, gr = self.goal
        done_p, done_r = self._wrist(joints[:, world])
        rel_p = gaussian_filter1d(gr[:frames].inv().apply(done_p - gp[:frames]), 2.0, axis=0, mode="nearest")
        rel_q = (gr[:frames].inv() * done_r).as_quat()
        for t in range(1, len(rel_q)):
            if rel_q[t] @ rel_q[t - 1] < 0:
                rel_q[t] = -rel_q[t]
        rel_q = gaussian_filter1d(rel_q, 2.0, axis=0, mode="nearest")
        rel_r = R.from_quat(rel_q / np.linalg.norm(rel_q, axis=1, keepdims=True))
        fingers = gaussian_filter1d(joints[:, world][:, hand.finger_q_ids], 2.0, axis=0, mode="nearest")
        cur_p, cur_r = self._wrist(q)
        upto = max(frontier, 1)
        fade = np.clip((upto - np.arange(n)) / 10.0, 0.0, 1.0)
        for gain in (0.3, 0.6, 1.0):
            g = gain * fade[:, None]
            f = np.clip(np.arange(n) - 1, 0, frames - 1)
            goal_p, goal_r = gp[f], gr[f]
            new_p = cur_p + g * (goal_p + goal_r.apply(rel_p[f]) - cur_p)
            delta = (goal_r * rel_r[f] * cur_r.inv()).as_rotvec()
            new_r = R.from_rotvec(g * delta) * cur_r
            c = self._with_wrist(q, new_p, new_r)
            c[:, hand.finger_q_ids] = q[:, hand.finger_q_ids] + g * (fingers[f] - q[:, hand.finger_q_ids])
            out.append(("copy", c))
        # Grip: shift the hand on the pen for the whole clip, and close or open the fingers.
        spare = max(self.count - 3, 0)
        for _ in range(spare // 2):
            dp, dr = rng.normal(0, 0.003, 3), rng.normal(0, np.radians(4.0), 3)
            f = np.clip(np.arange(n) - 1, 0, frames - 1)
            goal_p, goal_r = gp[f], gr[f]
            new_p = goal_p + goal_r.apply(R.from_rotvec(dr).apply(goal_r.inv().apply(cur_p - goal_p)) + dp)
            new_r = goal_r * R.from_rotvec(dr) * goal_r.inv() * cur_r
            c = self._with_wrist(q, new_p, new_r)
            c[:, hand.finger_q_ids] = q[:, hand.finger_q_ids] + rng.normal(0.01, 0.03)
            out.append(("grip", c))
        # Local: correct the wrist by the pen's error just before it went wrong.
        x = poses[:, world]
        err_p = gp[:frames] - x[:, :3]
        err_r = (gr[:frames] * R.from_quat(np.where(np.linalg.norm(x[:, 3:7], axis=1, keepdims=True) > 1e-6, x[:, 3:7], [0, 0, 0, 1.0])).inv()).as_rotvec()
        for _ in range(spare - spare // 2):
            f0 = int(np.clip(frontier - rng.integers(0, 12), 0, frames - 1))
            width, gain = float(rng.uniform(4.0, 15.0)), float(rng.uniform(0.3, 1.0))
            g = np.exp(-0.5 * ((np.arange(n) - f0) / width) ** 2)[:, None]
            dp = np.clip(gain * err_p[f0], -0.02, 0.02) + rng.normal(0, 0.002, 3)
            dr = gain * np.nan_to_num(err_r[f0])
            dr = dr * min(1.0, np.radians(20.0) / max(np.linalg.norm(dr), 1e-9))
            out.append(("local", self._with_wrist(q, cur_p + g * dp, R.from_rotvec(g * dr) * cur_r)))
        return out

    def _publish(self, step):
        path = self.output / f"reference_{self.version:04d}.npz"
        checksum = save_snapshot(path, self.binding, self.setup.scene.layout, provenance=dict(
            mode="practice", version=self.version, environment_steps=int(step),
            source_parquet=self.config["task"]["parquet"], object_contact_sha256=self.command))
        self.config["task"]["robot_reference_path"] = str(path)
        self.config["task"]["robot_reference_sha256"] = checksum
        self.metadata[CONFIG_METADATA_KEY] = json.dumps(self.config, sort_keys=True)
        self.metadata["robot_reference_version"] = str(self.version)

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
        jax.block_until_ready(learner)
        started = time.perf_counter()
        seed = self.training.seed + int(step)
        voc = float(env.env.voc_scale.numpy()[0])
        before, poses, joints = self.rollout(self.binding, learner, seed, voc)
        record = dict(step=int(step), voc=voc, version_before=self.version, before=before, candidates=[], accepted=False)
        print(f"[practice] step={step} voc={voc} v{self.version} kept={before['kept']:.2f} frontier={before['frontier']} tip={before['tip_cm']:.2f}cm on_board={before['on_board']:.2f} passed={before['passed']:.2f}", flush=True)
        best = None
        for kind, q in self.propose(before, poses, joints):
            if not self.in_bounds(q):
                record["candidates"].append(dict(kind=kind, rejected="out of bounds"))
                continue
            binding = self.bind(q)
            result, _, _ = self.rollout(binding, learner, seed, voc)
            record["candidates"].append(dict(result, kind=kind))
            if self.better(result, before) and (best is None or self.score(result) > self.score(best[0])):
                best = (result, kind, binding)
        if best is not None:
            # A fresh seed guards against keeping a lucky rollout.
            control, _, _ = self.rollout(self.binding, learner, seed + 100000, voc)
            check, _, _ = self.rollout(best[2], learner, seed + 100000, voc)
            record["validation"] = dict(control=control, candidate=check, kind=best[1])
            if self.better(check, control):
                self.binding = best[2]
                self.version += 1
                self._publish(step)
                record["accepted"] = True
        record.update(version_after=self.version, seconds=time.perf_counter() - started, upper_transitions=self.upper_transitions)
        if command_digest(self.setup.reference) != self.command:
            raise RuntimeError("upper update changed the pen goals")
        with (self.output.parent / "upper_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"[practice] step={step} accepted={record['accepted']} kind={best[1] if best else None} version={self.version} seconds={record['seconds']:.0f}", flush=True)
        if not record["accepted"]:
            return None
        replacement = RLEnv(replace(self.setup.scene, robot_reference=self.binding), self.setup.reference, config=deepcopy(env.env.config))
        replacement.reset()
        if replacement.device.is_cuda:
            replacement.capture_step()
        replacement.reset()
        return WarpRLEnv(replacement, publish_diagnostics=False, publish_step_metrics=False)
