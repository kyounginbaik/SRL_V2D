"""Practice in the Batch: bilevel practice whose outer level runs inside the RL training batch.

Lower level: FlashSAC trains one policy on all worlds (object-only reward, randomized friction).
Upper level: the reference holds K hand strategies end to end (PIB_SEG control frames each); worlds reset into
    random slots, so every training episode is a trial of some strategy. No separate rollouts are run.
Grasp first: probes are scored on the first H frames only (H starts at 1 s and grows 1.5x once the best strategy holds
    in half its probes), so practice first finds a grasp that holds, then extends it over the motion.
Fitness of a slot comes only from probe worlds (every 8th world): they always start at the slot's first frame, with the
    exact reference pose and no assistance. Fitness = (share of the clip kept + share of full clips) x share of writing
    frames with the tip within 3 cm, times (1 - lambda * distance from the demo). Lambda drops once a slot succeeds.
Every `interval` steps, once enough slots have evidence, the 2 weakest are replaced by mutants of the top 4
    (copy what the policy did, a local wrist fix before the failure, or a grip re-seat). Slots are re-ordered best first.
    Strategies that already keep 30% of the clip get a share of resets just before their failure point.
The environment is rebuilt with the new reference, the replay buffer is kept (object-only reward).
The best slot is exported as an ordinary single-clip reference, so the standard evaluation scores it unchanged.
Pen goals are never edited.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.retargeting.manipulation_bilevel import LocalProjector
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from flash_chord.training.environment import WarpRLEnv

LOST_TIP = 0.03
CAP_P, CAP_R, CAP_F = 0.10, np.radians(60.0), 0.8  # sanity limits only; the demo pull is the soft penalty


class PracticeInBatch:
    keep_replay = True

    def __init__(self, *, options, setup, embodiment, collision, training, environment_config, checkpoint_metadata, output_dir):
        o = dict(options)
        self.options, self.training, self.setup, self.metadata = o, training, setup, checkpoint_metadata
        self.config = deepcopy(environment_config)
        self.K, self.SEG, self.CLIP = int(o["slots"]), int(o["seg"]), int(o["clip"])
        self.interval, self.next_update = int(o["interval"]), 0
        self.replace_count, self.fail_share = int(o.get("replace_count", 2)), float(o.get("fail_reset", 0.25))
        self.probe, self.min_episodes, self.practice_from = int(o.get("probe", 8)), int(o.get("min_episodes", 6)), float(o.get("practice_from", 0.3))
        # Grasp first: probes are scored on keeping the object for the first H frames; H grows once the best strategy holds.
        self.H = int(o.get("horizon0", 40))
        self.grow, self.grow_at = float(o.get("grow", 1.5)), float(o.get("grow_at", 0.5))
        self.lam_before, self.lam_after = float(o.get("lam", 0.1)), float(o.get("lam_after", 0.02))
        self.rng = np.random.default_rng(training.seed + 20000)
        self.out = Path(output_dir)
        (self.out / "references").mkdir(parents=True, exist_ok=True)
        tiled = setup.scene.robot_reference
        self.total = tiled.num_frames
        if not self.K * self.SEG <= self.total < self.K * self.SEG + 8:
            raise ValueError(f"tiled reference has {tiled.num_frames} frames, expected {self.K} x {self.SEG} plus a short tail")
        self.projector = LocalProjector(embodiment, tiled)
        self.hand = next(h for h in self.projector.layout.hands if h.side == "right")
        lo, hi = self.projector.model.joint_limit_lower.numpy(), self.projector.model.joint_limit_upper.numpy()
        self.finger_limits = np.stack([lo[list(self.hand.finger_dof_ids)], hi[list(self.hand.finger_dof_ids)]], 1)
        self.tiled_target = np.asarray(tiled.joint_target, dtype=float)
        names = list(setup.reference.finger_joint_names("right"))
        fids = list(self.hand.finger_q_ids)
        flexing = ("FE", "PIP", "DIP", "IP")
        self.finger_groups = [[fids[i] for i, nm in enumerate(names) if f"_{f}_" in nm and any(x in nm for x in flexing)]
                              for f in ("thumb", "index", "middle", "ring", "pinky")]
        self.mode = str(o.get("mutations", "grip"))  # "grip": 11 numbers per strategy (grip offset + per-finger squeeze)
        self.demo = np.asarray(tiled.joint_q[: self.SEG], dtype=float)
        self.nq = tiled.num_joint_q
        # Goal (pen) at the control rate, from the first slot; every slot has the same goal.
        gp = np.asarray(setup.reference.object_body_pos_w(), dtype=float)[: self.SEG, 0]
        gq = np.asarray(setup.reference.object_body_quat_w(), dtype=float)[: self.SEG, 0]
        self.gp, self.gr = gp, R.from_quat(gq[:, [1, 2, 3, 0]])
        board = json.loads(Path(o["board"]).read_text())
        self.tip_local, normal, surface = np.array(board["tip_local"]), np.array(board["normal"]), np.array(board["surface_point"])
        self.gtip = self.gr.apply(self.tip_local) + gp
        h = (self.gtip - surface) @ normal
        on = h < np.percentile(h[: self.CLIP], 20) + 0.002
        on[self.CLIP:] = False
        idx = np.flatnonzero(on)
        on[: idx[0]] = False
        on[idx[-1] + 1:] = False
        self.on = on
        # The single-clip setup, for exporting the best slot as an ordinary reference.
        scene = self.config["scene"]
        env_cfg = instantiate_typed(OmegaConf.create(self.config["env"]), RLEnvConfig)
        self.single = setup_scene(
            parquet=o["single_parquet"], control_fps=env_cfg.sim.fps, motion_speed=self.config["task"]["motion_speed"],
            embodiment=embodiment, collision=collision, world_count=1, include_support=scene["support"],
            decompose_objects=scene["decompose_objects"], contact_friction=scene.get("contact_friction", 1.0),
            object_free_joint_damping=scene.get("object_free_joint_damping", 0.0),
        )
        if self.single.scene.robot_reference.num_frames != self.CLIP:
            raise ValueError(f"single clip has {self.single.scene.robot_reference.num_frames} frames, expected {self.CLIP}")
        self.single_target = np.asarray(self.single.scene.robot_reference.joint_target, dtype=float)
        # Population: slot 0 is the demo, the rest start as grip re-seats of it.
        self.slots = [self.demo.copy()] + [self._grip(self.demo, 1.0) for _ in range(self.K - 1)]
        self.lineage = ["demo"] + ["grip"] * (self.K - 1)
        self.lam, self.version, self.prev_t, self.tick = self.lam_before, 0, None, 0
        self._reset_stats(range(self.K))
        print(f"[pib] K={self.K} seg={self.SEG} clip={self.CLIP} writing frames={int(self.on.sum())}", flush=True)

    # ---------------- statistics from training episodes ----------------
    def _reset_stats(self, slots):
        if not hasattr(self, "E"):
            K, T, nf = self.K, self.SEG, len(self.hand.finger_q_ids)
            self.E, self.KS, self.SC = np.zeros(K), np.zeros(K), np.zeros(K)  # probe episodes, frames kept, full clips
            self.FH, self.B, self.N, self.tip = (np.zeros((K, T)) for _ in range(4))  # probe failure frames and tip quality
            self.perr, self.relp = np.zeros((K, T, 3)), np.zeros((K, T, 3))  # what the policy did, all worlds
            self.relq, self.fing, self.CN = np.zeros((K, T, 4)), np.zeros((K, T, nf)), np.zeros((K, T))
        for k in slots:
            for a in self._stat_arrays():
                a[k] = 0.0

    def _stat_arrays(self):
        return (self.E, self.KS, self.SC, self.FH, self.B, self.N, self.tip, self.perr, self.relp, self.relq, self.fing, self.CN)

    def after_training_step(self, env, step):
        """Selection evidence comes only from probe worlds: they start at a slot's first frame with no assistance."""
        e = env.env
        t = e.timestep.numpy().copy()
        if self.prev_t is None:
            self.prev_t = t
            return
        W = len(t)
        probe = (np.arange(W) % self.probe) == 0
        term = np.asarray(step.terminated).astype(bool).reshape(-1)
        trunc = np.asarray(step.truncated).astype(bool).reshape(-1)
        p = self.prev_t
        k, f = p // self.SEG, p % self.SEG
        inslot = k < self.K
        reached = probe & inslot & (f == self.H - 1) & (t == p + 1) & ~term & ~trunc
        if reached.any():
            np.add.at(self.E, k[reached], 1.0)
            np.add.at(self.KS, k[reached], float(self.H))
            np.add.at(self.SC, k[reached], 1.0)
        ended = probe & inslot & (term | trunc | (t != p + 1)) & (f < self.H - 1)
        if ended.any():
            ke, fe = k[ended], np.minimum(f[ended], self.H)
            np.add.at(self.E, ke, 1.0)
            np.add.at(self.KS, ke, fe)
            np.add.at(self.SC, ke, (fe >= self.H - 1).astype(float))
            failed = term[ended] & (fe < self.H - 1)
            np.add.at(self.FH, (ke[failed], fe[failed]), 1.0)
        adv = inslot & (f < self.CLIP - 1) & (t == p + 1) & ~term & ~trunc
        self.tick += 1
        if self.tick % 2 == 0 and adv.any():
            ids = e.command.body_ids_w.numpy().reshape(W, -1)[:, 0]
            bq = e.state_0.body_q.numpy()[ids][adv]
            w = np.flatnonzero(adv)
            kk, ff = k[adv], f[adv] + 1
            rot = R.from_quat(bq[:, 3:7])
            te = np.linalg.norm(rot.apply(self.tip_local) + bq[:, :3] - self.gtip[ff], axis=1)
            pr = probe[adv]
            np.add.at(self.N, (kk[pr], ff[pr]), 1.0)
            np.add.at(self.tip, (kk[pr], ff[pr]), te[pr])
            np.add.at(self.B, (kk[pr], ff[pr]), ((te > LOST_TIP) & self.on[ff])[pr])
            np.add.at(self.CN, (kk, ff), 1.0)
            np.add.at(self.perr, (kk, ff), self.gp[ff] - bq[:, :3])
            q = e.state_0.joint_q.numpy().reshape(W, -1)[w, : self.nq].astype(float)
            hp, hr = q[:, self.hand.wrist_pos_q_ids], R.from_quat(q[:, self.hand.wrist_orient_q_id: self.hand.wrist_orient_q_id + 4])
            g = self.gr[ff]
            np.add.at(self.relp, (kk, ff), g.inv().apply(hp - self.gp[ff]))
            rq = (g.inv() * hr).as_quat()
            rq *= np.sign(rq[:, 3:4] + 1e-9)
            np.add.at(self.relq, (kk, ff), rq)
            np.add.at(self.fing, (kk, ff), q[:, self.hand.finger_q_ids])
        self.prev_t = t

    def fitness(self):
        """Unassisted, from the first frame: share of the clip kept, plus share of full clips, times tip quality, minus the demo pull."""
        E = np.maximum(self.E, 1.0)
        kept = self.KS / E / self.H  # share of the current horizon kept, unassisted, from the first frame
        succ = self.SC / E  # share of probes that held through the whole horizon
        on = self.on[: self.SEG]
        on = on & (np.arange(self.SEG) < self.H)
        good = np.where(self.N[:, on].sum(1) > 0, 1.0 - self.B[:, on].sum(1) / np.maximum(self.N[:, on].sum(1), 1.0), 1.0)
        tip = self.tip[:, on].sum(1) / np.maximum(self.N[:, on].sum(1), 1.0)
        dist = np.array([self._demo_distance(s) for s in self.slots])
        task = succ * good
        # The demo pull is relative (a few percent), so it breaks ties but never outweighs task evidence.
        fit = np.where(self.E >= self.min_episodes, (kept * good + task) * np.clip(1.0 - self.lam * dist, 0.5, 1.0), np.nan)
        cum = np.cumsum(self.FH, axis=1)
        frontier = np.array([int(np.searchsorted(c, c[-1] / 2)) if c[-1] > 0 else self.CLIP - 1 for c in cum])
        return dict(task=task, fit=fit, kept=kept, surv=succ, good=good, tip_cm=tip * 100, dist=dist, samples=self.E.copy(), peak=frontier)

    # ---------------- strategies ----------------
    def _wrist(self, q):
        return q[:, self.hand.wrist_pos_q_ids].astype(float), R.from_quat(q[:, self.hand.wrist_orient_q_id: self.hand.wrist_orient_q_id + 4].astype(float))

    def _with_wrist(self, q, p, r):
        q = q.copy()
        q[:, self.hand.wrist_pos_q_ids] = p
        q[:, self.hand.wrist_orient_q_id: self.hand.wrist_orient_q_id + 4] = r.as_quat()
        return q

    def _demo_distance(self, q):
        p0, r0 = self._wrist(self.demo[: self.CLIP])
        p1, r1 = self._wrist(q[: self.CLIP])
        return float(np.linalg.norm(p1 - p0, axis=1).mean() / 0.05 + (r1 * r0.inv()).magnitude().mean() / np.radians(40.0))

    def _ok(self, q):
        p0, r0 = self._wrist(self.demo[: self.CLIP])
        p1, r1 = self._wrist(q[: self.CLIP])
        fq = self.hand.finger_q_ids
        return (np.linalg.norm(p1 - p0, axis=1).max() <= CAP_P and (r1 * r0.inv()).magnitude().max() <= CAP_R
                and np.abs(q[: self.CLIP, fq] - self.demo[: self.CLIP, fq]).max() <= CAP_F)

    def _finish(self, q):
        q = q.copy()
        fq = self.hand.finger_q_ids
        q[:, fq] = np.clip(q[:, fq], self.finger_limits[:, 0], self.finger_limits[:, 1])
        q[self.CLIP:] = q[self.CLIP - 1]
        return q

    def _grip(self, q, scale):
        """Re-seat the hand on the pen for the whole clip, and close or open the fingers."""
        rng, n = self.rng, self.SEG
        dp, dr = rng.normal(0, 0.006 * scale, 3), rng.normal(0, np.radians(8.0) * scale, 3)
        cp, cr = self._wrist(q)
        p = self.gp + self.gr.apply(R.from_rotvec(dr).apply(self.gr.inv().apply(cp - self.gp)) + dp)
        r = self.gr * R.from_rotvec(dr) * self.gr.inv() * cr
        c = self._with_wrist(q, p, r)
        for group in self.finger_groups:  # one squeeze per finger: close (+) or open (-) its bending joints together
            c[:, group] += rng.normal(0.0, 0.05 * scale)
        return self._finish(c)

    def _local(self, q, k, f0):
        rng = self.rng
        cnt = max(self.CN[k, f0], 1.0)
        err = self.perr[k, f0] / cnt
        width, gain = rng.uniform(4.0, 15.0), rng.uniform(0.3, 1.0)
        g = np.exp(-0.5 * ((np.arange(self.SEG) - f0) / width) ** 2)[:, None]
        cp, cr = self._wrist(q)
        return self._finish(self._with_wrist(q, cp + g * np.clip(gain * err, -0.02, 0.02), cr))

    def _copy(self, q, k, gain):
        """Pull the reference toward what the policy's hand actually did on this slot, where there is data."""
        cnt = self.CN[k][:, None]
        have = (cnt[:, 0] >= 20)
        if have.sum() < 20:
            return None
        relp = gaussian_filter1d(self.relp[k] / np.maximum(cnt, 1), 2.0, axis=0, mode="nearest")
        relq = gaussian_filter1d(self.relq[k] / np.maximum(cnt, 1), 2.0, axis=0, mode="nearest")
        fing = gaussian_filter1d(self.fing[k] / np.maximum(cnt, 1), 2.0, axis=0, mode="nearest")
        relq[~have] = [0, 0, 0, 1.0]
        relr = R.from_quat(relq / np.maximum(np.linalg.norm(relq, axis=1, keepdims=True), 1e-9))
        g = gain * gaussian_filter1d(have.astype(float), 3.0)[:, None]
        cp, cr = self._wrist(q)
        p = cp + g * (self.gp + self.gr.apply(relp) - cp)
        r = R.from_rotvec(g * (self.gr * relr * cr.inv()).as_rotvec()) * cr
        c = self._with_wrist(q, p, r)
        fq = self.hand.finger_q_ids
        c[:, fq] = np.where(have[:, None], q[:, fq] + g * (fing - q[:, fq]), q[:, fq])
        return self._finish(c)

    def mutate(self, parent, k, peak):
        if self.mode == "grip":
            for _ in range(20):
                big = self.rng.random() < 0.3
                c = self._grip(parent, 2.5 if big else 0.6)
                if self._ok(c):
                    return c, "regrip" if big else "grip"
            return self._grip(parent, 0.3), "grip"
        for _ in range(20):
            u = self.rng.random()
            if u < 0.35:
                c, kind = self._copy(parent, k, float(self.rng.choice([0.3, 0.6, 1.0]))), "copy"
            elif u < 0.65:
                c, kind = self._local(parent, k, int(np.clip(peak - self.rng.integers(0, 12), 0, self.CLIP - 1))), "local"
            else:
                big = self.rng.random() < 0.3
                c, kind = self._grip(parent, 2.5 if big else 1.0), "regrip" if big else "grip"
            if c is not None and self._ok(c):
                return c, kind
        return self._grip(parent, 1.0), "grip"

    # ---------------- binding, export, environment ----------------
    def _bind(self, q, target, fps):
        target = target.copy()
        p, r = self._wrist(q)
        target[:, self.hand.wrist_pos_dof_ids] = p
        target[:, self.hand.wrist_orient_dof_ids] = r.as_rotvec()
        target[:, self.hand.finger_dof_ids] = q[:, self.hand.finger_q_ids]
        return build_robot_reference_binding(self.projector.builder, self.projector.layout, q.astype(np.float32), target.astype(np.float32), fps)

    def _publish(self, step):
        fps = self.setup.scene.robot_reference.fps
        best = self._bind(self.slots[0][: self.CLIP], self.single_target, fps)
        path = self.out / "references" / f"best_{self.version:04d}.npz"
        checksum = save_snapshot(path, best, self.single.scene.layout, provenance=dict(
            mode="practice_in_batch", version=self.version, environment_steps=int(step),
            source_parquet=self.options["single_parquet"], object_contact_sha256=command_digest(self.single.reference)))
        self.config["task"]["parquet"] = self.options["single_parquet"]
        self.config["task"]["robot_reference_path"] = str(path)
        self.config["task"]["robot_reference_sha256"] = checksum
        self.metadata[CONFIG_METADATA_KEY] = json.dumps(self.config, sort_keys=True)
        self.metadata["robot_reference_version"] = str(self.version)

    def _environment(self, env):
        fps = self.setup.scene.robot_reference.fps
        q = np.concatenate(self.slots + [np.repeat(self.slots[-1][-1:], self.total - self.K * self.SEG, 0)], 0)
        tiled = self._bind(q, self.tiled_target, fps)
        new = RLEnv(replace(self.setup.scene, robot_reference=tiled), self.setup.reference, config=deepcopy(env.env.config))
        new.reset()
        if new.device.is_cuda:
            new.capture_step()
        new.reset()
        return new

    def _practice_frames(self, env, peaks, kepts):
        """Practice the hard part only for strategies that already get through part of the clip unassisted."""
        frames = [k * self.SEG + f for k, (pk, kv) in enumerate(zip(peaks, kepts)) if kv >= self.practice_from
                  for f in range(max(0, int(pk) - 12), max(1, int(pk) - 1))]
        env.reset_policy.set_practice_frames(frames, self.fail_share if frames else 0.0)

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        started = time.perf_counter()
        first = self.next_update == 0
        self.next_update = step + self.interval
        record = dict(step=int(step), version_before=self.version, lam=self.lam, horizon=self.H)
        if not first:
            m = self.fitness()
            fit = m["fit"]
            ready = np.flatnonzero(np.isfinite(fit))
            ranked = list(ready[np.argsort(-fit[ready])])
            waiting = [i for i in range(self.K) if i not in ranked]
            record.update({k: np.round(v[ranked + waiting], 4).tolist() for k, v in m.items()}, lineage=[self.lineage[i] for i in ranked + waiting])
            n_new = min(self.replace_count, max(0, len(ranked) - 4))
            if n_new == 0:
                # Not enough unassisted evidence yet: keep every strategy, only refresh what to practice.
                self._practice_frames(env.env, m["peak"], m["kept"])
                record.update(changed=False, ready=len(ranked))
                with (self.out / "practice_batch.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"[pib] step={step} waiting for evidence: {len(ranked)}/{self.K} strategies have {self.min_episodes}+ probe episodes", flush=True)
                return None
            # Keep everything except the weakest few ranked strategies; mutate the top four.
            keep = ranked[: len(ranked) - n_new] + waiting
            parents = ranked[:4]
            new_slots, new_lineage, fresh, perm = [self.slots[i] for i in keep], [self.lineage[i] for i in keep], [], list(keep)
            for _ in range(n_new):
                pk = int(self.rng.choice(parents))
                c, kind = self.mutate(self.slots[pk], pk, int(m["peak"][pk]))
                new_slots.append(c)
                new_lineage.append(f"{kind}<{self.lineage[pk].split('<')[0]}")
                fresh.append(len(new_slots) - 1)
                perm.append(pk)
            # Re-order evidence with the slots; old evidence is halved so the population tracks the improving policy.
            for a in self._stat_arrays():
                a[:] = a[perm] * 0.5
            self._reset_stats(fresh)
            peaks = [int(m["peak"][i]) for i in perm]
            kepts = [float(m["kept"][i]) if i not in fresh else 0.0 for i in perm]
            kepts = [0.0 if n in fresh else kv for n, kv in enumerate(kepts)]
            self.slots, self.lineage = new_slots, new_lineage
            best = ranked[0]
            if m["task"][best] > 0.5 and self.H >= self.CLIP - 1:
                self.lam = self.lam_after
            if m["surv"][best] >= self.grow_at and self.H < self.CLIP - 1:
                self.H = min(self.CLIP - 1, int(self.H * self.grow) + 1)
                self._reset_stats(range(self.K))  # evidence for the old horizon no longer applies
                record.update(horizon_grew=self.H)
            record.update(changed=True, best_task=float(m["task"][best]), best_kept=float(m["kept"][best]), best_lineage=self.lineage[0])
        else:
            peaks, kepts = [0] * self.K, [0.0] * self.K
        self.version += 1
        self._publish(step)
        replacement = self._environment(env)
        self._practice_frames(replacement, peaks, kepts)
        self.prev_t = None
        record.update(version_after=self.version, seconds=time.perf_counter() - started)
        with (self.out / "practice_batch.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if not first:
            print(f"[pib] step={step} H={self.H} best kept={record['best_kept']:.3f} held={record['surv'][0]:.3f} tip={record['tip_cm'][0]:.2f}cm "
                  f"lam={self.lam} best={self.lineage[0]} seconds={record['seconds']:.0f}", flush=True)
        return WarpRLEnv(replacement, publish_diagnostics=False, publish_step_metrics=False)
