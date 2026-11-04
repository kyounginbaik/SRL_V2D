"""UniRetarget upper level on YAM + Sharpa, called between FlashSAC training blocks.

The reference is the demonstrated palm motion expressed relative to the object goal,
plus edits: one grasp offset for the contact segment, local palm corrections and a
finger closure change. Every candidate goes through arm IK under the speed limit, runs
with the frozen policy and no assistance, and is published only if a fresh seed confirms
it beats the current reference. Object commands are never edited.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
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

TOL_P, TOL_R = 0.03, np.radians(30.0)
LOST_P, LOST_R = 0.15, np.radians(150.0)
SLIP_P, SLIP_R = 0.02, np.radians(20.0)
# Bounds around the demonstration, so the reference stays a refinement of it.
GRASP_P, GRASP_R = 0.04, np.radians(30.0)
LOCAL_P, LOCAL_R = 0.03, np.radians(25.0)
FINGER = 0.3


def identity_edit():
    return dict(dp=np.zeros(3), dr=np.zeros(3), close=0.0, local=[])


def measure(poses, palm, onset):
    """Median failure over worlds: first frame out of budget, what the object did by then."""
    worlds = poses.shape[1]
    frontier, slip, types, cm, deg, lift = [], [], [], [], [], []
    for w in range(worlds):
        x = poses[:, w, 0]
        i = x[0]
        al = R.from_quat(i[3:7]) * R.from_quat(i[10:14]).inv()
        gp = al.apply(x[:, 7:10]) + i[:3] - al.apply(i[7:10])
        gr = al * R.from_quat(x[:, 10:14])
        ro = R.from_quat(x[:, 3:7])
        ep = np.linalg.norm(x[:, :3] - gp, axis=1)
        er = (ro.inv() * gr).magnitude()
        total = len(ep) + 1
        n = np.arange(1, len(ep) + 1)
        limit = 0.7 * n + 0.3 * total
        spent = (np.cumsum(ep) > TOL_P * limit) | (np.cumsum(er) > TOL_R * limit)
        lost = (ep > LOST_P) | (er > LOST_R)
        lost[: min(onset + 40, len(lost))] = False  # The arm may still be catching up after closing.
        over = spent | lost
        f = int(np.flatnonzero(over)[0]) if over.any() else len(ep)
        pp = palm[:, w, :3]
        # A palm with an invalid pose has diverged: its world fails from that frame on.
        pq = np.array(palm[:, w, 3:7], dtype=float)
        bad = ~np.isfinite(pq).all(axis=1) | (np.linalg.norm(np.nan_to_num(pq), axis=1) < 1e-6)
        if bad.any():
            f = min(f, int(np.flatnonzero(bad)[0]))
            pq[bad] = (0.0, 0.0, 0.0, 1.0)
            print(f"[measure] world {w}: {int(bad.sum())} invalid palm poses from frame {int(np.flatnonzero(bad)[0])}", flush=True)
        pr = R.from_quat(pq)
        rel_p = pr.inv().apply(x[:, :3] - pp)
        rel_r = pr.inv() * ro
        start = min(onset, len(ep) - 1)
        moved = np.linalg.norm(rel_p - rel_p[start], axis=1) > SLIP_P
        turned = (rel_r * rel_r[start].inv()).magnitude() > SLIP_R
        late = np.flatnonzero((moved | turned)[start:])
        s = start + int(late[0]) if len(late) else -1
        goal_moved = np.linalg.norm(gp[: max(f, 1)] - gp[0], axis=1).max()
        moved_obj = np.linalg.norm(x[: max(f, 1), :3] - x[0, :3], axis=1).max()
        travel = np.linalg.norm(x[: min(f + 25, len(x)), :3] - x[0, :3], axis=1).max()
        wanted = max(np.linalg.norm(gp[: min(f + 25, len(gp))] - gp[0], axis=1).max(), 1e-6)
        lift.append(float(min(travel / wanted, 1.0)))
        kind = "held" if f == len(ep) else ("never_held" if goal_moved > 0.05 and moved_obj < 0.4 * goal_moved else ("slips" if 0 <= s <= f else "off_track"))
        frontier.append(f)
        slip.append(s)
        types.append(kind)
        cm.append(100 * ep.mean())
        deg.append(np.degrees(er.mean()))
    values, counts = np.unique(types, return_counts=True)
    cm, deg, frontier = np.asarray(cm), np.asarray(deg), np.asarray(frontier)
    passed = (frontier >= poses.shape[0]) & (cm < 100 * TOL_P) & (deg < np.degrees(TOL_R))
    return dict(
        frontier=int(np.median(frontier)),
        frontier_min=int(frontier.min()),
        slip=int(np.median(slip)),
        kind=str(values[np.argmax(counts)]),
        kinds={str(k): int(c) for k, c in zip(values, counts)},
        cm=float(cm.mean()),
        deg=float(deg.mean()),
        lift=float(np.median(lift)),
        passed=float(passed.mean()),
    )


def score(m):
    """Larger is better: full passes first, then how far it gets, then accuracy."""
    return (m["passed"], m["frontier"], m.get("lift", 0.0), -max(m["cm"] / 3.0, m["deg"] / 30.0))


def better(a, b):
    """A real gain only: publishing clears the replay buffer, so tiny differences are not worth it."""
    if a["passed"] != b["passed"]:
        return a["passed"] >= b["passed"] + 0.1
    if a["frontier"] != b["frontier"]:
        return a["frontier"] >= b["frontier"] + 3
    # A grasp that moves the object further is real progress even before the frontier moves.
    if a.get("lift", 0.0) >= b.get("lift", 0.0) + 0.05 and a["cm"] <= b["cm"] + 1.0:
        return True
    if a.get("lift", 0.0) <= b.get("lift", 0.0) - 0.05:
        return False
    return max(a["cm"] / 3.0, a["deg"] / 30.0) <= 0.95 * max(b["cm"] / 3.0, b["deg"] / 30.0)


class YamReferenceEditor:
    def __init__(self, *, options, setup, embodiment, collision, training, environment_config, checkpoint_metadata, output_dir):
        self.options = dict(options)
        if training.resume:
            raise ValueError("resume is not supported for the YAM upper level")
        self.interval = int(self.options["interval"])
        # A grasp already proven in the physics search should be the reference from the first step: training on a
        # reference that never touches the object only teaches the policy to ignore it.
        self.next_update = 0 if self.options.get("pool") else int(self.options.get("warmup", self.interval))
        self.count = int(self.options.get("candidates", 12))
        self.training, self.setup, self.metadata = training, setup, checkpoint_metadata
        self.config = deepcopy(environment_config)
        self.output = Path(output_dir) / "references"
        self.output.mkdir(parents=True, exist_ok=True)
        self.baseline = setup.scene.robot_reference
        self.binding = self.baseline
        self.command = command_digest(setup.reference)
        self.projector = LocalProjector(embodiment, self.baseline)
        self.names = list(self.projector.layout.scalar_joints.names)
        self.rng = np.random.default_rng(training.seed + 10000)
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
        sys.path.insert(0, self.options["code_dir"])
        import yam_loop as Y

        self.Y = Y
        Y.ONSET_FLOOR = float(self.options.get("onset_floor", 2.0))
        self.side = self.options["side"]
        self.task = Y.YamTask(self.options["episode"], self.side, int(self.options.get("object", 0)))
        self.task.set_stage("real")
        self.onset = self.task.onset
        # Contact schedule: every object carried must pass, each judged against whichever hand holds it at that frame; the worst decides.
        self.holds = {}
        meta = Path(self.options["episode"]) / "recon_meta.json"
        environ = __import__("os").environ
        if meta.is_file() and not environ.get("RECON_ONE_HOLD"):
            all_holds = json.loads(meta.read_text()).get("holds", [])
            upto = int(environ.get("RECON_HOLDS_UPTO", 0))
            used = all_holds[:upto] if upto else all_holds
            for h in used:
                self.holds.setdefault(int(h["index"]), []).append(h)
            # Every object in the scene counts, including one no hand is scheduled to hold.
            meta_objects = json.loads(meta.read_text()).get("objects", [])
            self.unheld = [k for k in range(len(meta_objects)) if k not in {int(h["index"]) for h in all_holds}]
            if len(all_holds) < 2 and not self.unheld:
                self.holds = {}
            self.hold_onset = {k: Y.YamTask(self.options["episode"], v[0]["side"], k).onset for k, v in self.holds.items()}
            self.hold_cut = {k: (max(int(x.get("release") or x["object_end"]) for x in v) if upto else max(int(x["object_end"]) for x in v)) for k, v in self.holds.items()}
            # A hold the next hand joins without a release is only judged until that hand is on the object.
            self.judged_until_join = set()
            for k, v in self.holds.items():
                joins = [x for x in all_holds[upto:] if int(x["index"]) == k] if upto else []
                if joins and v[-1].get("release") is None:
                    self.hold_cut[k] = min(self.hold_cut[k], max(int(joins[0]["onset"]), int(v[-1]["onset"])) + 20)
                    self.judged_until_join.add(k)
        prefix = "R" if self.side == "right" else "L"
        qids = self.projector.layout.scalar_joints.q_ids
        self.arm_q = [qids[self.names.index(f"{prefix}_arm_j{i}")] for i in range(1, 7)]
        self.finger_names = [n for n in self.names if n.startswith(self.side + "_")]
        self.finger_q = [qids[self.names.index(n)] for n in self.finger_names]
        self.limits = np.array([self.task.m.jnt_range[self.task.m.joint(f"{self.side}_arm_{n}").id] for n in self.finger_names])
        self.palm_body = next(h for h in self.projector.layout.hands if h.side == self.side).palm_frame.body_id
        self.human_q = self.baseline.joint_q.copy()
        self.human_p, self.human_r = self._palm_path(self.human_q)
        self.edit = identity_edit()
        self.actor = self.policy = None
        self.version = 0
        # Whole-grasp alternatives from the physics search; local edits alone cannot leave the demonstrated grasp.
        self.pool = []
        for parquet in self.options.get("pool", ()):
            other = setup_scene(
                parquet=parquet, control_fps=self.eval_config.sim.fps, motion_speed=task["motion_speed"],
                embodiment=embodiment, collision=collision, world_count=1,
                include_support=scene["support"], decompose_objects=scene["decompose_objects"],
                contact_friction=scene.get("contact_friction", 1.0),
                object_free_joint_damping=scene.get("object_free_joint_damping", 0.0),
                source_frame_playback=task["source_frame_playback"],
                motion_start_frame=task["motion_start_frame"], motion_end_frame=task["motion_end_frame"],
            )
            if command_digest(other.reference) != self.command:
                raise ValueError(f"pool reference {parquet} has different object commands")
            self.pool.append((str(parquet), other.scene.robot_reference))
        self._publish(0)

    def _palm_path(self, q):
        t, d = self.task, self.task.p
        p, r = [], []
        for f in range(len(q)):
            d.qpos[:] = t.base_q
            d.qpos[t.q_arm] = q[f, self.arm_q]
            __import__("mujoco").mj_kinematics(t.m, d)
            a, b = t.pose(d)
            p.append(a)
            r.append(b)
        return np.asarray(p), R.concatenate(r)

    def build(self, edit):
        """Reference from the human motion and one edit; arm joints respect the speed limit by construction."""
        t, Y = self.task, self.Y
        n = len(self.human_q)
        gp, gr = t.gp_raw[:n], t.gr_raw[:n]
        contact = np.clip((np.arange(n) - (self.onset - 20)) / 20.0, 0.0, 1.0)
        rel_p = gr.inv().apply(self.human_p - gp) + contact[:, None] * edit["dp"]
        rel_r = R.from_rotvec(contact[:, None] * edit["dr"]) * (gr.inv() * self.human_r)
        p = gp + gr.apply(rel_p)
        r = gr * rel_r
        frames = np.arange(n)
        for e in edit["local"]:
            g = np.exp(-0.5 * ((frames - e["frame"]) / e["width"]) ** 2)[:, None]
            p = p + g * np.asarray(e["dp"])
            r = R.from_rotvec(g * np.asarray(e["dr"])) * r
        q = self.human_q.copy()
        arm = Y.READY.copy()
        step = Y.ARM_SPEED / Y.FPS * 0.9
        t.p.qpos[:] = t.base_q
        for f in range(n):
            arm, _, _ = t.arm_ik(arm, p[f], r[f], steps=8, limit=step)
            q[f, self.arm_q] = arm
        grasp = self.human_q[min(self.onset, n - 1), self.finger_q] - self.human_q[0, self.finger_q]
        closing = np.where(grasp >= 0.0, self.limits[:, 1], self.limits[:, 0])
        fingers = self.human_q[:, self.finger_q] + contact[:, None] * edit["close"] * (closing - self.human_q[:, self.finger_q])
        q[:, self.finger_q] = np.clip(fingers, self.limits[:, 0], self.limits[:, 1])
        target = self.baseline.joint_target.copy()
        target[:, self.projector.dids] = q[:, self.projector.qids]
        return build_robot_reference_binding(self.projector.builder, self.projector.layout, q, target, self.baseline.fps)

    def rollout(self, binding, learner, seed, voc_probe_scale=0.0):
        """Frozen policy, full episode, from the task's own start. No assistance unless a VOC probe asks for it."""
        eval_config = deepcopy(self.eval_config)
        if voc_probe_scale > 0.0:
            eval_config.voc = deepcopy(self.train_voc)
            eval_config.voc_scale = float(voc_probe_scale)
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
        self._action_dim = adapter.action_dim
        W = env.world_count
        per_world = env.model.body_count // W
        poses, palm, palms = [], [], []
        for _ in range(binding.num_frames - 1):
            action, key = self.policy(state, obs, key)
            obs = jnp.array(adapter.step(action).observation, copy=True)
            jax.block_until_ready(obs)
            b = env.state_0.body_q.numpy()
            x = b[env.command.body_ids_w.numpy().reshape(-1)].reshape(W, -1, 7)
            gp = env.command.body_target_pos_w.numpy().reshape(W, -1, 3)
            gq = env.command.body_target_quat_w.numpy().reshape(W, -1, 4)
            poses.append(np.concatenate([x, gp, gq], axis=-1))
            palm.append(b.reshape(W, per_world, 7)[:, self.palm_body])
            palms.append(b.reshape(W, per_world, 7))
        # Judge the object this reference moves, not whichever body happens to come first.
        everything = np.asarray(poses)
        poses = everything[:, :, [int(self.options.get("object", 0))]]
        m = measure(poses, np.asarray(palm), self.onset)
        if self.holds:
            palms = np.asarray(palms)
            hands = {h.side: h.palm_frame.body_id for h in self.projector.layout.hands}
            parts = []
            for k, v in self.holds.items():
                n = min(self.hold_cut[k] + 1, len(everything))
                stitched = palms[:n, :, hands[v[0]["side"]]].copy()
                for h in v[1:]:
                    stitched[int(h["start"]):] = palms[int(h["start"]):n, :, hands[h["side"]]]
                parts.append(measure(everything[:n, :, [k]], stitched, self.hold_onset[k]))
                # Cut mid-motion, it cannot have finished the move yet: whether it is on track is what counts.
                if k in getattr(self, "judged_until_join", ()):
                    parts[-1]["lift"] = 1.0
            # An object no hand holds only has to stay on its recorded path.
            for k in getattr(self, "unheld", ()):
                parts.append(measure(everything[:, :, [k]], palms[:, :, next(iter(hands.values()))], 0))
                parts[-1]["lift"] = 1.0
            m = dict(min(parts, key=lambda x: (x["passed"], -x["cm"])), holds=[{a: x[a] for a in ("passed", "cm", "deg", "kinds")} for x in parts])
        return m, poses

    def propose(self, m, poses):
        """Edits chosen by the failure type; physics decides which survive."""
        out = []
        rng = self.rng
        frontier = min(m["frontier"], poses.shape[0] - 1)
        cause = m["slip"] if m["kind"] == "slips" and m["slip"] >= 0 else frontier
        x = poses[:, 0, 0]
        i = x[0]
        al = R.from_quat(i[3:7]) * R.from_quat(i[10:14]).inv()
        gp = al.apply(x[:, 7:10]) + i[:3] - al.apply(i[7:10])
        for c in range(self.count):
            e = deepcopy(self.edit)
            grasp = m["kind"] == "never_held" or (m["kind"] == "slips" and c % 2 == 0)
            if grasp:
                e["dp"] = np.clip(e["dp"] + rng.normal(0, 0.012, 3), -GRASP_P, GRASP_P)
                turn = R.from_rotvec(rng.normal(0, np.radians(10.0), 3)) * R.from_rotvec(e["dr"])
                v = turn.as_rotvec()
                e["dr"] = v * min(1.0, GRASP_R / max(np.linalg.norm(v), 1e-9))
                e["close"] = float(np.clip(e["close"] + rng.normal(0.03, 0.05), -FINGER, FINGER))
            else:
                # Spread over moments before the detected point instead of betting on one frame.
                f = int(np.clip(cause - rng.integers(0, 12), self.onset, poses.shape[0] - 1))
                gain = float(rng.uniform(0.3, 1.0))
                dp = np.clip(gain * (gp[f] - x[f, :3]) + rng.normal(0, 0.004, 3), -LOCAL_P, LOCAL_P)
                dr = gain * (al * R.from_quat(x[f, 10:14]) * R.from_quat(x[f, 3:7]).inv()).as_rotvec() + rng.normal(0, 0.03, 3)
                dr = dr * min(1.0, LOCAL_R / max(np.linalg.norm(dr), 1e-9))
                e["local"] = e["local"][-30:] + [dict(frame=f, width=float(rng.uniform(3.0, 10.0)), dp=dp.tolist(), dr=dr.tolist())]
            out.append(e)
        return out

    def _publish(self, step):
        path = self.output / f"reference_{self.version:04d}.npz"
        checksum = save_snapshot(path, self.binding, self.setup.scene.layout, provenance=dict(
            mode="yam_reference", version=self.version, environment_steps=int(step),
            source_parquet=self.config["task"]["parquet"], object_contact_sha256=self.command))
        edit = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.edit.items()}
        (self.output / f"edit_{self.version:04d}.json").write_text(json.dumps(edit, indent=1) + "\n")
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
        before, poses = self.rollout(self.binding, learner, seed)
        record = dict(step=int(step), version_before=self.version, baseline=before, candidates=[], accepted=False)
        print(f"[upper] step={step} v{self.version} {before['kind']} frontier={before['frontier']} slip={before['slip']} {before['cm']:.2f}cm {before['deg']:.1f}deg passed={before['passed']:.2f}", flush=True)
        best = None
        for edit in self.propose(before, poses):
            candidate = self.build(edit)
            result, _ = self.rollout(candidate, learner, seed)
            record["candidates"].append(result)
            # An edit that still holds nothing is no reason to throw away the replay buffer, or a grasp known to work.
            if result["kind"] == "never_held" and self.pool:
                continue
            if better(result, before) and (best is None or score(result) > score(best[0])):
                best = (result, edit, candidate)
        for name, candidate in self.pool:
            if name == getattr(self, "current_pool", None) and self.version > 0 and not any(self.edit["local"]) and np.all(self.edit["dp"] == 0):
                continue
            result, _ = self.rollout(candidate, learner, seed)
            result["pool"] = name
            record["candidates"].append(result)
            # A whole new grasp must already work with the current policy, not merely beat a failure.
            if result["passed"] >= float(self.options.get("pool_min_pass", 0.5)) and better(result, before) and (best is None or score(result) > score(best[0])):
                best = (result, dict(identity_edit(), pool=name), candidate)
            # A policy trained on a reference that never touches the object is no judge of one that does: while nothing is held, also test the grasp on its own.
            already = name == getattr(self, "current_pool", None) and not any(self.edit["local"]) and np.all(self.edit["dp"] == 0)
            # On a partial-gate run the same test applies to a reference that passes nothing, however it fails.
            partial_gate = float(self.options.get("alone_min", 0.3)) == 0 and before["passed"] == 0
            if (before["kind"] == "never_held" or partial_gate) and not already:
                trained = self.policy
                self.policy = lambda state_, o, k: (jnp.zeros((o.shape[0], self._action_dim), dtype=o.dtype), k)
                alone, _ = self.rollout(candidate, learner, self.training.seed)
                self.policy = trained
                result["alone"] = {a: alone[a] for a in ("passed", "cm", "deg", "kinds", "lift")}
                carried = sum(v for a, v in alone["kinds"].items() if a in ("held", "off_track")) / max(1, sum(alone["kinds"].values()))
                # A partly working grasp is still far better than a reference that never touches the object, and the policy can firm it up.
                if carried >= float(self.options.get("alone_min", 0.3)) and alone["lift"] >= 0.8 and (best is None or score(alone) > score(best[0])):
                    best = (alone, dict(identity_edit(), pool=name, alone=True), candidate)
        if best is not None:
            # A fresh seed guards against keeping a lucky rollout.
            control, _ = self.rollout(self.binding, learner, seed + 100000)
            if best[1].get("alone"):
                trained = self.policy
                self.policy = lambda state_, o, k: (jnp.zeros((o.shape[0], self._action_dim), dtype=o.dtype), k)
                check, _ = self.rollout(best[2], learner, seed + 100000)
                self.policy = trained
            else:
                check, _ = self.rollout(best[2], learner, seed + 100000)
            record["validation"] = dict(control=control, candidate=check)
            # A grasp that carries the object beats a reference that never touches it, even if few worlds pass yet.
            rescued = best[1].get("alone") and control["kind"] == "never_held" and check["frontier"] >= control["frontier"] + 3 and check["lift"] >= 0.8
            if better(check, control) or rescued:
                self.edit, self.binding = best[1], best[2]
                adopted = self.edit.get("pool")
                self.version += 1
                self._publish(step)
                if adopted:
                    # Later local edits refine the adopted grasp. It stays in the pool, so a later
                    # collapse can return to it; only the reference in use is skipped.
                    self.current_pool = adopted
                    self.human_q = self.binding.joint_q.copy()
                    self.human_p, self.human_r = self._palm_path(self.human_q)
                    self.edit = identity_edit()
                record["accepted"] = True
        record.update(version_after=self.version, seconds=time.perf_counter() - started)
        if command_digest(self.setup.reference) != self.command:
            raise RuntimeError("upper update changed the object commands")
        with (self.output.parent / "upper_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"[upper] step={step} accepted={record['accepted']} version={self.version} seconds={record['seconds']:.0f}", flush=True)
        if not record["accepted"]:
            return None
        replacement = RLEnv(replace(self.setup.scene, robot_reference=self.binding), self.setup.reference, config=deepcopy(env.env.config))
        replacement.reset()
        if replacement.device.is_cuda:
            replacement.capture_step()
        replacement.reset()
        return WarpRLEnv(replacement, publish_diagnostics=False, publish_step_metrics=False)
