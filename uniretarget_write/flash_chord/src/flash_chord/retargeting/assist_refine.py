"""Assistance-weighted reference refinement (UniRetarget upper level, no IK in the loop).

Lower level: FlashSAC follows the current hand reference (hand-tracking reward on) and tracks the object; the virtual
    object controller (VOC) assists with strength s per world, decaying over the curriculum.
Signal: under partial assistance (s <= s_max) the object sags unless the hand carries it, so a small object error means
    the hand state physically supports the object. Every world-frame sample gets weight w = exp(-err / sigma).
Upper level, every `interval` steps, directly in joint space:
    ref(t) <- ref(t) + eta * clip( sum_w w * (executed hand(t) - ref(t)) / sum_w w )   (hand pose in the goal object frame)
    with a trust region per update, temporal smoothing, and a small pull back toward the demo.
Cost: running sums over the training batch; no rollouts, no search, no population.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv
from flash_chord.retargeting.manipulation_bilevel import LocalProjector
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from flash_chord.training.environment import WarpRLEnv


class AssistRefine:
    keep_replay = True

    def __init__(self, *, options, setup, embodiment, collision, training, environment_config, checkpoint_metadata, output_dir):
        o = dict(options)
        self.o, self.training, self.setup, self.metadata = o, training, setup, checkpoint_metadata
        self.config = deepcopy(environment_config)
        self.interval, self.next_update = int(o.get("interval", 5000192)), int(o.get("warmup", 5000192))
        self.eta = float(o.get("eta", 0.5))  # step toward the weighted executed state
        self.sigma = float(o.get("sigma", 0.01))  # object error scale [m]
        self.s_max = float(o.get("s_max", 0.75))  # only partially assisted samples carry information
        self.max_p, self.max_r, self.max_f = float(o.get("max_p", 0.005)), np.radians(float(o.get("max_r_deg", 3.0))), float(o.get("max_f", 0.05))
        self.pull = float(o.get("pull", 0.02))  # pull toward the demo per update
        self.min_w = float(o.get("min_w", 20.0))
        self.baseline = bool(int(o.get("baseline", 0)))  # 1: move by (weighted mean - plain mean) of executed states, cancelling the shared tracking lag
        self.out = Path(output_dir)
        (self.out / "references").mkdir(parents=True, exist_ok=True)
        ref = setup.scene.robot_reference
        self.binding, self.fps = ref, ref.fps
        self.projector = LocalProjector(embodiment, ref)
        self.hand = next(h for h in self.projector.layout.hands if h.side == "right")
        lo, hi = self.projector.model.joint_limit_lower.numpy(), self.projector.model.joint_limit_upper.numpy()
        self.finger_limits = np.stack([lo[list(self.hand.finger_dof_ids)], hi[list(self.hand.finger_dof_ids)]], 1)
        self.target = np.asarray(ref.joint_target, dtype=float)
        self.demo = np.asarray(ref.joint_q, dtype=float).copy()
        self.q = self.demo.copy()
        self.T, self.nq = len(self.q), ref.num_joint_q
        gp = np.asarray(setup.reference.object_body_pos_w(), dtype=float)[: self.T, 0]
        gq = np.asarray(setup.reference.object_body_quat_w(), dtype=float)[: self.T, 0]
        self.gp, self.gr = gp, R.from_quat(gq[:, [1, 2, 3, 0]])
        self.version, self.tick = 0, 0
        self._clear()
        self._publish(0)
        print(f"[assist] frames={self.T} eta={self.eta} sigma={self.sigma} s_max={self.s_max}", flush=True)

    def _clear(self):
        nf = len(self.hand.finger_q_ids)
        self.W = np.zeros(self.T)
        self.P = np.zeros((self.T, 3)); self.Qr = np.zeros((self.T, 4)); self.F = np.zeros((self.T, nf))
        self.err_sum, self.err_n = np.zeros(self.T), np.zeros(self.T)
        self.N = np.zeros(self.T); self.P0 = np.zeros((self.T, 3)); self.Q0 = np.zeros((self.T, 4)); self.F0 = np.zeros((self.T, nf))

    def _wrist(self, q):
        h = self.hand
        return q[:, h.wrist_pos_q_ids].astype(float), R.from_quat(q[:, h.wrist_orient_q_id: h.wrist_orient_q_id + 4].astype(float))

    def after_training_step(self, env, step):
        self.tick += 1
        if self.tick % 2:
            return
        e = env.env
        W = e.world_count
        t = e.timestep.numpy()
        s = e.reset_policy.applied_voc_scale.numpy()
        live = (t < self.T) & (s <= self.s_max)
        if not live.any():
            return
        ids = e.command.body_ids_w.numpy().reshape(W, -1)[:, 0]
        bq = e.state_0.body_q.numpy()[ids][live]
        ft = t[live]
        err = np.linalg.norm(bq[:, :3] - self.gp[ft], axis=1)
        err = err + 0.02 * (R.from_quat(bq[:, 3:7]) * self.gr[ft].inv()).magnitude()  # 1 rad counts as 2 cm
        w = np.exp(-err / self.sigma)
        q = e.state_0.joint_q.numpy().reshape(W, -1)[live, : self.nq].astype(float)
        hp, hr = q[:, self.hand.wrist_pos_q_ids], R.from_quat(q[:, self.hand.wrist_orient_q_id: self.hand.wrist_orient_q_id + 4])
        g = self.gr[ft]
        rq = (g.inv() * hr).as_quat()
        rq *= np.sign(rq[:, 3:4] + 1e-9)
        np.add.at(self.W, ft, w)
        np.add.at(self.P, ft, w[:, None] * g.inv().apply(hp - self.gp[ft]))
        np.add.at(self.Qr, ft, w[:, None] * rq)
        np.add.at(self.F, ft, w[:, None] * q[:, self.hand.finger_q_ids])
        np.add.at(self.err_sum, ft, err)
        np.add.at(self.err_n, ft, 1.0)
        np.add.at(self.N, ft, 1.0)
        np.add.at(self.P0, ft, g.inv().apply(hp - self.gp[ft]))
        np.add.at(self.Q0, ft, rq)
        np.add.at(self.F0, ft, q[:, self.hand.finger_q_ids])

    def _publish(self, step):
        path = self.out / "references" / f"reference_{self.version:04d}.npz"
        checksum = save_snapshot(path, self.binding, self.setup.scene.layout, provenance=dict(
            mode="assist_refine", version=self.version, environment_steps=int(step),
            source_parquet=self.config["task"]["parquet"], object_contact_sha256=command_digest(self.setup.reference)))
        self.config["task"]["robot_reference_path"] = str(path)
        self.config["task"]["robot_reference_sha256"] = checksum
        self.metadata[CONFIG_METADATA_KEY] = json.dumps(self.config, sort_keys=True)
        self.metadata["robot_reference_version"] = str(self.version)

    def _bind(self, q):
        target = self.target.copy()
        p, r = self._wrist(q)
        target[:, self.hand.wrist_pos_dof_ids] = p
        target[:, self.hand.wrist_orient_dof_ids] = r.as_rotvec()
        target[:, self.hand.finger_dof_ids] = q[:, self.hand.finger_q_ids]
        return build_robot_reference_binding(self.projector.builder, self.projector.layout, q.astype(np.float32), target.astype(np.float32), self.fps)

    def update(self):
        """One upper-level step: move the reference toward the weighted executed hand state, inside a trust region."""
        have = self.W >= self.min_w
        if have.sum() < 10:
            return None
        Wd = np.maximum(self.W, 1e-9)[:, None]
        rel_p = self.P / Wd
        rq = self.Qr / Wd
        rq[~have] = [0, 0, 0, 1.0]
        rel_r = R.from_quat(rq / np.maximum(np.linalg.norm(rq, axis=1, keepdims=True), 1e-9))
        fing = self.F / Wd
        cur_p, cur_r = self._wrist(self.q)
        # desired executed wrist in world, from the goal frame
        want_p = self.gp + self.gr.apply(rel_p)
        want_r = self.gr * rel_r
        if self.baseline:  # advantage form: where the good samples sit relative to all samples, not relative to the reference
            Nd = np.maximum(self.N, 1e-9)[:, None]
            q0 = self.Q0 / Nd; q0[~have] = [0, 0, 0, 1.0]
            r0 = R.from_quat(q0 / np.maximum(np.linalg.norm(q0, axis=1, keepdims=True), 1e-9))
            dp = self.gr.apply(rel_p - self.P0 / Nd)
            dr = (self.gr * rel_r * (self.gr * r0).inv()).as_rotvec()
            df = fing - self.F0 / Nd
        else:
            dp = want_p - cur_p
            dr = (want_r * cur_r.inv()).as_rotvec()
            df = fing - self.q[:, self.hand.finger_q_ids]
        mask = have.astype(float)
        conf = mask * np.clip(self.W / (self.W[have].mean() + 1e-9), 0.0, 1.0)  # frames with little evidence move less
        for a in (dp, dr, df):
            a *= conf[:, None]
            a[:] = gaussian_filter1d(a, 3.0, axis=0, mode="nearest")
        dp = self.eta * dp; dr = self.eta * dr; df = self.eta * df
        n = np.linalg.norm(dp, axis=1, keepdims=True); dp *= np.minimum(1.0, self.max_p / np.maximum(n, 1e-9))
        n = np.linalg.norm(dr, axis=1, keepdims=True); dr *= np.minimum(1.0, self.max_r / np.maximum(n, 1e-9))
        df = np.clip(df, -self.max_f, self.max_f)
        # small pull back toward the demo, so the reference stays a refinement of it
        dp_demo, dr_demo = self._wrist(self.demo)
        new_p = cur_p + dp + self.pull * (dp_demo - cur_p)
        new_r = R.from_rotvec(dr) * cur_r
        new_r = R.from_rotvec(self.pull * (dr_demo * new_r.inv()).as_rotvec()) * new_r
        q = self.q.copy()
        h = self.hand
        q[:, h.wrist_pos_q_ids] = new_p
        q[:, h.wrist_orient_q_id: h.wrist_orient_q_id + 4] = new_r.as_quat()
        fq = list(h.finger_q_ids)
        q[:, fq] = np.clip(q[:, fq] + df + self.pull * (self.demo[:, fq] - q[:, fq]), self.finger_limits[:, 0], self.finger_limits[:, 1])
        stats = dict(frames_with_evidence=int(have.sum()), mean_weight=float(self.W[have].mean()),
                     obj_err_cm=float((self.err_sum.sum() / max(self.err_n.sum(), 1)) * 100),
                     step_wrist_mm=float(np.linalg.norm(dp, axis=1).max() * 1000), step_finger=float(np.abs(df).max()),
                     drift_wrist_cm=float(np.linalg.norm(new_p - dp_demo, axis=1).max() * 100))
        return q, stats

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update = step + self.interval
        started = time.perf_counter()
        res = self.update()
        record = dict(step=int(step), version_before=self.version)
        if res is None:
            record.update(skipped="not enough partially assisted evidence")
            self._clear()
            with (self.out / "assist_updates.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            return None
        self.q, stats = res
        self.binding = self._bind(self.q)
        self.version += 1
        self._publish(step)
        new = RLEnv(replace(self.setup.scene, robot_reference=self.binding), self.setup.reference, config=deepcopy(env.env.config))
        new.reset()
        if new.device.is_cuda:
            new.capture_step()
        new.reset()
        self._clear()
        record.update(stats, version_after=self.version, seconds=time.perf_counter() - started)
        with (self.out / "assist_updates.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[assist] step={step} v{self.version} " + " ".join(f"{k}={v:.3g}" for k, v in stats.items()), flush=True)
        return WarpRLEnv(new, publish_diagnostics=False, publish_step_metrics=False)
