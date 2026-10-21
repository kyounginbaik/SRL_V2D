"""Batched contextual-bandit search: one distinct candidate per physics world."""
from copy import deepcopy
from dataclasses import replace
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from flash_chord.envs.rl import RLEnv
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.learner import Transition
from flash_chord.training.flash_sac.checkpoint import save_learner_checkpoint
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor, upper_cost, passes_study_gate, contact_proximity_gate, ProposalRejected, contact_directed_action
from flash_chord.retargeting.manipulation_bilevel import assert_device_binding
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.retargeting.parallel_rollout import rollout_candidates


class ParallelReferenceEditor(TTSAReferenceEditor):
    def __init__(self, **kwargs):
        options = dict(kwargs['options'])
        worlds = int(options.get('eval_worlds', 2048))
        if worlds != int(options.get('upper_trials', worlds)):
            raise ValueError('parallel search requires one candidate per evaluation world')
        if worlds != kwargs['setup'].scene.world_count:
            raise ValueError('training and candidate evaluation must use the same world count')
        if options.get('study_variant') != 'chunk' or options.get('penetration_aware', False):
            raise ValueError('parallel editor supports chunk/contact scoring without penetration scoring')
        # Single-world baseline/confirmation, not four duplicate replicas.
        kwargs['options'] = {**options, 'eval_worlds': 1}
        super().__init__(**kwargs)
        self.options = options
        self.upper_worlds = worlds
        self.parallel_scene = self.setup.scene
        self.parallel_reference = self.setup.reference
        self.parallel_config = deepcopy(self.eval_config)
        self.parallel_policy = None
        if options.get('edit_scope') == 'wrists_only':
            if self.joint_basis.shape[0] != 12:
                raise ValueError('Wrist-only editor must have 12 spatial channels')
            probe = self.candidate_joint_q(np.ones(self.basis.shape[1]*12, np.float32)*.5)
            for hand in self.projector.layout.hands:
                np.testing.assert_array_equal(probe[:,hand.finger_q_ids], self.baseline.joint_q[:,hand.finger_q_ids])
            print(f'[method] wrist-only upper action_dim={self.basis.shape[1]*12}; finger reference invariant', flush=True)
        manifest_path = self.output.parent/'ttsa_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(engine='parallel_flashsac', eval_worlds=worlds,
            candidate_worlds=worlds, replicas_per_candidate=1,
            upper_learning='one simultaneous proposal batch, then upper learner update',
            confirmation='deterministic rerun, not a randomization holdout',
            reset_mode='explicit exact reference pose; no sampled finger opening')
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    def _proposal_training_env(self, env):
        """Training env whose worlds track the current reference or this round's candidate references."""
        from flash_chord.retargeting.parallel_rollout import pack_candidate_references, install_candidate_references
        W = self.setup.scene.world_count
        cq, ct = self._proposal_pool
        n_prop = int(W * float(self.options.get('proposal_fraction', .5)))
        idx = np.arange(n_prop) % len(cq)
        qs = np.concatenate([np.repeat(self.binding.joint_q[None], W - n_prop, 0), np.asarray(cq)[idx]])
        ts = np.concatenate([np.repeat(self.binding.joint_target[None], W - n_prop, 0), np.asarray(ct)[idx]])
        train = RLEnv(replace(self.setup.scene, robot_reference=self.binding), self.setup.reference, config=deepcopy(env.env.config))
        train.reset()
        install_candidate_references(train, pack_candidate_references(train, qs, ts), allow_auto_reset=True)
        train._step_graph = None
        train.reset(); train.capture_step(); train.reset()
        wrapped = WarpRLEnv(train, publish_diagnostics=False, publish_step_metrics=False)
        wrapped.keep_replay = True
        print(f'[proposals] training env: {W - n_prop} worlds current reference v{self.version}, {n_prop} worlds on {len(cq)} candidate references', flush=True)
        return wrapped

    def _failure_window(self, learner, step):
        """Window [t_c, t_f] from the measured object error of the current reference (see tools/patch_failure_window.py)."""
        from flash_chord.retargeting.parallel_rollout import rollout_candidates as _rc
        W = self.upper_worlds
        bq, bt = self.binding.joint_q.copy(), self.binding.joint_target.copy()
        _rc(self, learner, np.repeat(bq[None], W, 0), np.repeat(bt[None], W, 0))
        self._offset_cached = True
        F = bq.shape[0]; fps = self.baseline.fps
        sig = np.nan_to_num(self._signed_object_offset.astype(np.float64), nan=9., posinf=9., neginf=-9.)   # (Fm, W, O, 3)
        finite = np.isfinite(self._signed_object_offset).all(axis=(0, 2, 3))
        stable = finite & (np.linalg.norm(sig, axis=-1).max(axis=(0, 2)) < .3)
        keep = stable if stable.sum() >= 16 else finite
        if keep.sum() < 4:
            print(f'[window] step={step} no usable worlds; unrestricted round', flush=True); return None
        rel = (sig - sig[:1])[:, keep]
        err_pos = np.median(np.linalg.norm(rel, axis=-1), axis=1).max(axis=1)                       # (Fm,)
        OP = np.nan_to_num(self._object_pose.astype(np.float64))[:, keep]
        def _safe(qq):
            n = np.linalg.norm(qq, axis=-1, keepdims=True); return np.where(n > 1e-8, qq / np.maximum(n, 1e-8), np.array([0., 0., 0., 1.]))
        Fm, Wk, O = OP.shape[0], OP.shape[1], OP.shape[2]
        E = (Rotation.from_quat(_safe(OP[..., 3:7]).reshape(-1, 4)) * Rotation.from_quat(_safe(OP[..., 10:14]).reshape(-1, 4)).inv()).as_quat().reshape(Fm, Wk, O, 4)
        Erel = (Rotation.from_quat(E.reshape(-1, 4)) * Rotation.from_quat(np.repeat(E[None, 0], Fm, 0).reshape(-1, 4)).inv()).magnitude().reshape(Fm, Wk, O)
        err_ori = np.median(Erel, axis=1).max(axis=1)
        bad = (err_pos > .03) | (err_ori > np.radians(float(self.options.get('window_ori_deg', 30.))))   # per-frame thresholds
        if not bad.any():
            print(f'[window] step={step} no frame over 3 cm / 30 deg (max {err_pos.max()*100:.1f} cm, {np.degrees(err_ori.max()):.0f} deg); unrestricted round', flush=True)
            return None
        t_f = int(np.argmax(bad)) + 1
        # where the miss STARTS: last frame before t_f with error still under 1 cm / 10 deg (the cause lies between here and t_f)
        low = (err_pos[:t_f - 1] <= .01) & (err_ori[:t_f - 1] <= np.radians(10))
        t_s = (int(np.flatnonzero(low)[-1]) + 1) + 1 if low.any() else 1
        if str(self.options.get('window_detect', 'error')) == 'motion':
            # reference object moved, physical object did not follow (median over kept worlds), over a 0.25 s lag
            L = max(1, int(round(.25 * fps)))
            P_act, P_ref = OP[..., 0:3], OP[:, 0, :, 7:10]                       # (Fm, Wk, O, 3), (Fm, O, 3)
            d_act = np.median(np.linalg.norm(P_act[L:] - P_act[:-L], axis=-1), axis=1)   # (Fm-L, O)
            d_ref = np.linalg.norm(P_ref[L:] - P_ref[:-L], axis=-1)                       # (Fm-L, O)
            mism = np.concatenate([np.zeros(L, bool), ((d_ref > .01) & (d_act < .5 * d_ref)).any(axis=1)])
            k = t_f - 1
            if 0 <= k < len(mism) and mism[max(0, k - L):k + 1].any():
                while k > 1 and mism[k - 1]:
                    k -= 1
                t_s = int(k) + 1
            else:
                k = t_f - 1
                while k > 1 and not mism[k]:
                    k -= 1
                if k > 1 and t_f - k <= 1.5 * fps:
                    while k > 1 and mism[k - 1]:
                        k -= 1
                    t_s = int(k) + 1
            self._win_detect = 'motion'
        onsets = []
        ref = self.setup.reference
        for side in ref.sides:
            act = np.asarray(ref.contact_active(side)).reshape(F, -1).any(axis=1)
            idx = np.flatnonzero(act)
            if len(idx):
                starts = [idx[0]] + [idx[i] for i in range(1, len(idx)) if idx[i] - idx[i - 1] > 11]
                onsets += [int(x) for x in starts]
        prior = [o for o in onsets if o <= t_f]
        t_c = max(prior) if prior else max(1, int(t_f - fps))
        a, b = max(1, int(min(t_c, t_s) - .25 * fps)), min(F - 2, int(t_f))   # window ENDS at the detection frame
        t = np.arange(F, dtype=float); ramp = max(1., .25 * fps)
        m = np.clip(np.minimum((t - a) / ramp + 1., (b - t) / ramp + 1.), 0., 1.)
        m = .5 - .5 * np.cos(np.pi * m); m[0] = 0.; m[-1] = 0.
        chunks = (self.basis.astype(np.float64) * m[:, None]).max(axis=0) > .3
        self._win_info = dict(detect=str(self.options.get('window_detect', 'error')), first_fail_frame=t_f, miss_start_frame=int(t_s), fail_pos_cm=round(float(err_pos[t_f - 1]) * 100, 2), fail_ori_deg=round(float(np.degrees(err_ori[t_f - 1])), 1),
                              onset_frame=int(t_c), window=[int(a), int(b)], chunks=int(chunks.sum()), total_chunks=int(len(chunks)))
        print(f'[window] step={step} first_fail={t_f} ({self._win_info["fail_pos_cm"]} cm / {self._win_info["fail_ori_deg"]} deg) miss_start={t_s} onset={t_c} window=[{a},{b}] chunks={chunks.sum()}/{len(chunks)}', flush=True)
        self._win_chunks = chunks
        return m

    def _window_blend(self, q, m):
        """candidate = current reference outside the window (mask m in [0,1]); joint targets recomputed."""
        bq = self.binding.joint_q
        out = bq.copy()
        qids = np.asarray(self.projector.qids)
        out[:, qids] = bq[:, qids] + m[:, None] * (q[:, qids] - bq[:, qids])
        for qs, ds in self.projector.ball_joints:
            Rb = Rotation.from_quat(bq[:, qs:qs + 4]); Rc = Rotation.from_quat(q[:, qs:qs + 4])
            out[:, qs:qs + 4] = (Rotation.from_rotvec(m[:, None] * (Rc * Rb.inv()).as_rotvec()) * Rb).as_quat()
        out[0] = bq[0]; out[-1] = bq[-1]
        tt = self.baseline.joint_target.copy()
        tt[:, self.projector.dids] = out[:, self.projector.qids]
        for qi, di in self.projector.ball_joints:
            tt[:, di:di + 3] = Rotation.from_quat(out[:, qi:qi + 4]).as_rotvec()
        return out, tt

    def _ilc_candidates(self, learner):
        """Physics-feedback edits of the current reference: wrist translation, wrist rotation about the object,
        slip-driven squeeze (see tools/patch_ilc_upper.py and tools/patch_ilc_upper_v2.py)."""
        from scipy.ndimage import gaussian_filter1d
        W = self.upper_worlds
        bq, bt = self.binding.joint_q.copy(), self.binding.joint_target.copy()
        if not getattr(self, '_offset_cached', False):
            rollout_candidates(self, learner, np.repeat(bq[None], W, 0), np.repeat(bt[None], W, 0))
        self._offset_cached = False
        sig = np.nan_to_num(self._signed_object_offset.astype(np.float64), nan=9., posinf=9., neginf=-9.)
        keep = np.linalg.norm(sig, axis=-1).max(axis=(0, 2)) < .3
        self._ilc_stable_worlds = int(keep.sum())
        if keep.sum() < 16:
            self._ilc_feedback = dict(stable_worlds=int(keep.sum()), skipped=True)
            return []
        sig = sig - sig[:1]                                                       # remove each world's initial offset
        med = np.median(sig[:, keep], axis=1)                                              # (F-1, O, 3)
        OP = np.nan_to_num(self._object_pose.astype(np.float64)); PP = np.nan_to_num(self._palm_pose.astype(np.float64))
        F = bq.shape[0]; fps = self.baseline.fps; Fm = OP.shape[0]; O = OP.shape[2]
        def safe(qq):
            n = np.linalg.norm(qq, axis=-1, keepdims=True); return np.where(n > 1e-8, qq / np.maximum(n, 1e-8), np.array([0., 0., 0., 1.]))
        # orientation error rotvec per frame/world/object, initial removed, median over worlds
        E = (Rotation.from_quat(safe(OP[..., 3:7]).reshape(-1, 4)) * Rotation.from_quat(safe(OP[..., 10:14]).reshape(-1, 4)).inv())
        E = E.as_quat().reshape(Fm, W, O, 4)
        Erel = (Rotation.from_quat(E.reshape(-1, 4)) * Rotation.from_quat(np.repeat(E[None, 0], Fm, 0).reshape(-1, 4)).inv()).as_rotvec().reshape(Fm, W, O, 3)
        rmed = np.median(Erel[:, keep], axis=1)                                            # (F-1, O, 3)
        off = np.zeros((F, O, 3)); off[1:] = med; off = gaussian_filter1d(off, 1.5, axis=0)
        rot = np.zeros((F, O, 3)); rot[1:] = rmed; rot = gaussian_filter1d(rot, 1.5, axis=0)
        _n = np.linalg.norm(rot, axis=-1, keepdims=True); rot = rot * np.minimum(1., .5 / np.maximum(_n, 1e-9))  # cap 0.5 rad
        ref = self.setup.reference
        names = list(self.projector.layout.scalar_joints.names); qids = list(self.projector.qids)
        lo, hi = self.projector.lower, self.projector.upper
        hand_index = {h.side: i for i, h in enumerate(self.projector.layout.hands)}
        t = np.arange(F, dtype=float)
        smooth = lambda x: .5 - .5 * np.cos(np.pi * np.clip(x, 0, 1))
        bound = float(self.options.get('wrist_translation_offset_m', .08))
        tgt_pos = np.asarray(ref.object_body_pos_w(), np.float64)
        order = ['thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP', 'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP',
                 'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP', 'middle_DIP', 'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP',
                 'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP']
        flexj = [j for j in order if not (j.endswith('AA') or j in ('thumb_CMC_FE', 'pinky_CMC'))]
        per_side, slips = {}, {}
        for side in ref.sides:
            active = np.asarray(ref.contact_active(side)).reshape(F, -1).any(axis=1)
            parts = np.asarray(ref.contact_part_ids(side)).reshape(F, -1) - 1
            idx = np.flatnonzero(active)
            if not len(idx):
                continue
            runs, a0, prev = [], idx[0], idx[0]
            for i in idx[1:]:
                if i - prev > 11:
                    runs.append((a0, prev)); a0 = i
                prev = i
            runs.append((a0, prev))
            segs = []
            for a0, b0 in [(a0, b0) for a0, b0 in runs if b0 - a0 + 1 >= 5]:
                sg = parts[a0:b0 + 1][parts[a0:b0 + 1] >= 0]
                o = int(np.bincount(sg).argmax()) if len(sg) else 0
                start, end = min(F - 2, a0 + 6), min(F - 2, b0 + 6)
                w = np.zeros(F); w[(t >= start) & (t <= end)] = 1
                pre = t < start; w[pre] = smooth((t[pre] - (start - .3 * fps)) / (.3 * fps))
                post = t > end; w[post] = 1 - smooth((t[post] - end) / fps)
                # in-hand slip: object position in the palm frame vs just after closing
                h = hand_index[side]; k0 = min(Fm - 1, a0 + 10); k1 = min(Fm - 1, b0)
                Rp = Rotation.from_quat(safe(PP[k0:k1 + 1, :, h, 3:7]).reshape(-1, 4))
                rel = Rp.inv().apply((OP[k0:k1 + 1, :, o, :3] - PP[k0:k1 + 1, :, h, :3]).reshape(-1, 3)).reshape(k1 - k0 + 1, W, 3)
                S = float(np.median(np.linalg.norm(rel - rel[:1], axis=-1)[:, keep], axis=1).max()) if k1 > k0 else 0.
                segs.append(dict(o=o, start=start, end=end, w=w, post=post, slip=S))
            slips[side] = [round(sg['slip'], 4) for sg in segs]
            cols = [qids[names.index(f'{side}_wrist_{c}')] for c in 'xyz']
            import newton as _nt
            _b = self.projector.builder
            ball = [_b.joint_q_start[j] for j, kd in enumerate(_b.joint_type) if kd == _nt.JointType.BALL and side in str(_b.joint_label[j])][0]
            fcols = [qids[names.index(f'{side}_{j}')] for j in flexj]
            per_side[side] = (cols, ball, fcols, segs)
        tail = np.clip((F - 1 - t) / 3., 0, 1); tail[0] = 0
        out = []
        specs = self.options.get('ilc_specs') or [f'{g}:{m}:0:0' for g in self.options.get('ilc_gains', [.5, .75, 1., 1.25]) for m in self.options.get('ilc_modes', ['xy', 'xyz'])]
        for spec in specs:
            g_t, mode, g_r, g_s = str(spec).split(':'); g_t, g_r, g_s = float(g_t), float(g_r), float(g_s)
            q = bq.copy()
            for side, (cols, ball, fcols, segs) in per_side.items():
                dpos = np.zeros((F, 3)); drv = np.zeros((F, 3)); sq = np.zeros(F)
                for sg in segs:
                    w, o, end, post = sg['w'], sg['o'], sg['end'], sg['post']
                    d = -off[:, o] * w[:, None]; d[post] = -off[end, o] * w[post, None]
                    if mode == 'xy':
                        d[:, 2] = 0
                    r = -rot[:, o] * w[:, None]; r[post] = -rot[end, o] * w[post, None]
                    dpos += g_t * d; drv += g_r * r
                    sq = np.maximum(sq, g_s * .4 * min(1., sg['slip'] / .03) * w)
                dpos *= tail[:, None]; drv *= tail[:, None]; sq *= tail
                # rotation about the object: palm position moves on the circle around the (target) object position
                o_main = segs[0]['o'] if segs else 0
                if g_r:
                    Rd = Rotation.from_rotvec(drv)
                    c = tgt_pos[:, o_main]
                    q[:, cols] = c + Rd.apply(q[:, cols] - c)
                    q[:, ball:ball + 4] = (Rd * Rotation.from_quat(q[:, ball:ball + 4])).as_quat()
                dev = q[:, cols] + dpos - self.baseline.joint_q[:, cols]
                n = np.linalg.norm(dev, axis=1, keepdims=True)
                q[:, cols] = self.baseline.joint_q[:, cols] + dev * np.minimum(1., bound / np.maximum(n, 1e-9))
                if g_s:
                    for c in fcols:
                        j = qids.index(c)
                        q[:, c] = np.clip(q[:, c] + sq, lo[j], hi[j])
            q[0] = bq[0]; q[-1] = bq[-1]
            tt = self.baseline.joint_target.copy()
            tt[:, self.projector.dids] = q[:, self.projector.qids]
            for qi, di in self.projector.ball_joints:
                tt[:, di:di+3] = Rotation.from_quat(q[:, qi:qi+4]).as_rotvec()
            out.append((dict(spec=str(spec), max_change=float(np.abs(q - bq).max())), q, tt))
        self._ilc_feedback = dict(stable_worlds=int(keep.sum()), median_offset_norm_max_m=float(np.linalg.norm(med, axis=-1).max()),
                                  median_rotation_max_deg=float(np.degrees(np.linalg.norm(rmed, axis=-1).max())),
                                  in_hand_slip_m=slips)
        return out

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
        if getattr(self, '_skip_rounds', 0) > 0:
            self._skip_rounds -= 1; print(f'[hold] step={step} upper round skipped (reference frozen, {self._skip_rounds} more)', flush=True); return None
        self._last_record = None
        out = self._one_round(env, learner, step, stage)
        n = int(self.options.get('hold_refine_rounds', 0))
        rec = self._last_record or {}
        cand = (rec.get('validation') or {}).get('candidate') or {}
        if (n and not getattr(self, '_hold_done', False) and rec.get('accepted')
                and cand.get('object_position_robust_m', 1.) <= float(self.options.get('hold_position_m', .04))
                and cand.get('unstable_world_fraction', 1.) <= .1):
            self._hold_done = True
            print(f'[hold] step={step} grasp holds (robust pos {cand["object_position_robust_m"]:.3f} m); lower frozen for {n} upper rounds', flush=True)
            for i in range(1, n + 1):
                o = self._one_round(env, learner, step + i, stage)
                if o is not None:
                    out = o
            print(f'[hold] step={step} refine done; reference version {self.version}', flush=True)
            self._just_held = True
        stopped = self._stop_on_success(step, env, learner, stage)
        # keep the lower FROZEN and refine further instead of returning to lower training (which drifts): up to hold_blocks extra blocks
        b = 0
        while not stopped and getattr(self, '_hold_done', False) and b < int(self.options.get('hold_blocks', 0)):
            b += 1; print(f'[hold] step={step} extra hold block {b} (lower stays frozen)', flush=True)
            for i in range(1, n + 1):
                o = self._one_round(env, learner, step + 1000 * b + i, stage)
                if o is not None: out = o
            self._just_held = True
            stopped = self._stop_on_success(step, env, learner, stage)
        return out

    def _stop_on_success(self, step, env=None, learner=None, stage=None):
        thr = float(self.options.get('stop_success_fraction', 0.9))
        st = getattr(self, '_accepted_stats', None)
        if not thr or not st:
            return
        pos_ok = st.get('object_position_robust_m', 1.) <= float(self.options.get('stop_position_m', .03))
        hold_ok = (not self.options.get('stop_requires_hold', True)) or getattr(self, '_hold_done', False)
        # right after every hold block the confirmation (fresh-replica MP-SR) decides; otherwise the accepted candidate must look good first
        ok = hold_ok and (getattr(self, '_just_held', False) or (pos_ok and st.get('object_success_robust_fraction', 0.) >= thr)) and st.get('unstable_world_fraction', 1.) <= .1
        if ok and env is not None and self.options.get('stop_confirm', True):
            # CONFIRM on fresh worlds: the accepted candidate's stats are the best of ~120 (optimistic). One more round (lower frozen)
            # re-measures the accepted reference on its 128 fresh baseline replicas; stop only if THOSE pass.
            self._accepted_stats = None; self._last_record = None; _v_confirmed = int(self.version)
            self._one_round(env, learner, step + 500, stage)
            rb = (self._last_record or {}).get('replica_baseline') or {}
            _rows = getattr(self, '_last_base_rows', None) or []
            # MP-SR as the metric computes it: per world, errors AFTER aligning the first pose (here: first-frame offset removed),
            # episode means < 3 cm and < 30 deg; on the 128 baseline (= confirmed reference) worlds of the round's replica rollout
            NB = int(self.options.get('baseline_replicas', 128)); mpsr = 0.
            try:
                from scipy.spatial.transform import Rotation as _R
                OP = np.nan_to_num(np.asarray(self._object_pose, np.float64)[:, -NB:])                           # (Fm, NB, O, 14)
                q = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
                Fm, Wk, O = OP.shape[:3]; pos = np.zeros((Wk, O)); ori = np.zeros((Wk, O))
                for w in range(Wk):   # official Track 3 alignment (as tools/eval_fixed_reference.py): first pose of object 0
                    Ra = _R.from_quat(q(OP[0, w, 0, 10:14])) * _R.from_quat(q(OP[0, w, 0, 3:7])).inv(); off = OP[0, w, 0, 7:10] - Ra.apply(OP[0, w, 0, 0:3])
                    for o in range(O):
                        pw = Ra.apply(OP[:, w, o, 0:3]) + off; qw = Ra * _R.from_quat(q(OP[:, w, o, 3:7]))
                        pos[w, o] = np.linalg.norm(pw - OP[:, w, o, 7:10], axis=1).mean(); ori[w, o] = (qw * _R.from_quat(q(OP[:, w, o, 10:14])).inv()).magnitude().mean()
                mpsr = float(((pos < .03) & (ori < np.radians(30.))).all(axis=1).mean())
            except Exception as _e:
                print(f'[stop] MP-SR computation failed: {_e}', flush=True)
            rb = dict(rb, replica_mp_sr=round(mpsr, 3), replicas=len(_rows))
            print(f'[stop] step={step} fresh-replica MP-SR (episode-mean < 3 cm and < 30 deg): {mpsr:.3f} over {len(_rows)} replicas', flush=True)
            if mpsr >= .8:   # promising on the fresh replicas -> run the metric's own evaluation (official script, other seed, 64 worlds)
                import subprocess as _sp, sys as _sys, os as _os
                _T = self.output.parent; _root = _T.parents[5]
                _ck = _T / f'policy_{int(step)}.safetensors'; _rf = _T / 'references' / f'reference_{int(self.version if not (self._last_record or {}).get("accepted") else self.version - 1):04d}.npz'
                try:
                    if not _ck.exists():   # the runner saves this step's checkpoint only AFTER the upper call returns: save it now
                        import jax as _jax
                        from flash_chord.training.flash_sac.checkpoint import save_learner_checkpoint as _save
                        _save(_T, learner, metadata=dict(self.metadata or {}, environment_steps=int(_jax.device_get(learner.environment_steps)),
                              global_update_step=int(_jax.device_get(learner.global_update_step)), replay_included=False), save_replay=False)
                        _ck = _T / f'policy_{int(_jax.device_get(learner.environment_steps))}.safetensors'
                    _res = _sp.run([_sys.executable, str(_root / 'code/eval_fixed_reference.py'), str(_root), str(_rf), str(_ck)], capture_output=True, text=True,
                                   cwd='/workspace', env={**_os.environ, 'EVAL_SEED': '45', 'EVAL_WORLDS': '64'}, timeout=1800); _o = _res.stdout
                    if not any(x.startswith('policy_') for x in _o.splitlines()):
                        print('[stop] official evaluation produced no result; stderr tail:\n' + '\n'.join(_res.stderr.strip().splitlines()[-12:]), flush=True)
                    _l = [x for x in _o.splitlines() if x.startswith('policy_')][-1]; _d = json.loads(_l.split(' ', 1)[1])
                    mpsr = float(_d['aligned_mp_sr_pass_fraction']); rb = dict(rb, eval_mp_sr_seed45=mpsr, eval_median_pos_cm=_d['aligned_median_pos_cm'])
                    print(f'[stop] step={step} official evaluation (seed 45, 64 worlds) of policy_{int(step)} x {_rf.name}: MP-SR {mpsr:.3f}, median {_d["aligned_median_pos_cm"]} cm', flush=True)
                except Exception as _e:
                    print(f'[stop] official evaluation failed: {_e}', flush=True); mpsr = 0.
            ok = (mpsr >= float(self.options.get('confirm_mp_sr', .85)) and rb.get('object_position_robust_m', 1.) <= float(self.options.get('confirm_position_m', .05))
                  and rb.get('unstable_world_fraction', 1.) <= .1)   # judged on the reference measured by the fresh replicas (v before this round)
            print(f'[stop] step={step} confirmation on fresh replicas: pos {rb.get("object_position_robust_m", -1):.4f} m success {rb.get("object_success_robust_fraction", -1):.3f} -> {"confirmed" if ok else "NOT confirmed, continue"}', flush=True)
            if ok: st = dict(rb, confirmed_on_fresh_replicas=True)
        if ok:
            total = int(step) + int(self.training.world_count)
            _vc = int(locals().get('_v_confirmed', self.version))
            info = dict(step=int(step), reference_version=_vc, stats=st, rule=f'after hold-refine: robust pos <= stop_position_m, success fraction >= {thr}, unstable <= 0.1')
            (self.output.parent / 'success_pair.json').write_text(json.dumps(info, indent=1))
            object.__setattr__(self.training, 'total_environment_steps', total)
            print(f'[stop] step={step} success pair: policy_{step} x reference v{_vc} (success {st["object_success_robust_fraction"]:.3f}, pos {st.get("object_position_robust_m", -1):.4f} m); budget cut to {total}', flush=True)
            return True
        elif getattr(self, '_hold_done', False):
            if getattr(self, '_just_held', False):
                self._just_held = False; self._skip_rounds = int(self.options.get('post_hold_lower_rounds', 0))
                if self._skip_rounds: print(f'[hold] step={step} reference frozen for {self._skip_rounds} upper rounds (lower trains on it)', flush=True)
            self._hold_cooldown = getattr(self, '_hold_cooldown', 0)
            if self._hold_cooldown >= int(self.options.get('hold_retry_after_rounds', 2)):
                self._hold_done = False; self._hold_cooldown = 0; print(f'[hold] step={step} re-armed after cooldown', flush=True)
            else:
                self._hold_cooldown += 1

    def _one_round(self, env, learner, step, stage):
        self.upper_round += 1
        jax.block_until_ready(learner)
        started = time.perf_counter()
        previous_steps = self.validation_steps
        seed = self.training.seed + step
        baseline, _, _, feedback = self.rollout(self.binding, learner, seed, return_feedback=True)
        weight = float(self.options.get('wrench_weight', 1.)) if self.mode == 'reactor_wrench' else 0.
        def cost(metrics, q):
            return upper_cost(metrics, q-self.baseline.joint_q, self.baseline.fps, weight,
                              robust=True, contact_proximity_weight=self.contact_proximity_weight)
        def gate(metrics, control):
            return (metrics['all_errors_finite'] and metrics['full_horizon']
                and passes_study_gate(metrics, control)
                and (not self.contact_proximity_weight or contact_proximity_gate(metrics, control)))
        baseline_cost = cost(baseline, self.binding.joint_q)
        obs = jnp.repeat(self.observation(baseline, feedback), self.upper_worlds, axis=0)
        action, context = self.prepare_upper(obs)
        if self.guided_contact_proposal:
            guided = contact_directed_action(feedback['contact_direction_bins'],
                list(self.projector.layout.scalar_joints.names), self.projector.qids,
                self.joint_basis, float(self.options.get('wrist_translation_offset_m', .03)))
            action = action.at[0].set(jnp.asarray(guided))
        actions = np.asarray(action)
        _space = getattr(self, 'grasp_proto', None) or getattr(self, 'oc', None)
        if _space is not None:
            actions = _space.structured(actions, self.basis.shape[1]*self.joint_basis.shape[0])
            action = jnp.asarray(actions, dtype=action.dtype)
        # REPLICA_MODE
        R = int(self.options.get('replicas', 1)); NB = int(self.options.get('baseline_replicas', 0))
        if R > 1:
            W = self.upper_worlds; C = (W - NB) // R; NB = W - C * R
            wd = self.basis.shape[1]*self.joint_basis.shape[0]
            win_mask = self._failure_window(learner, step) if self.options.get('failure_window') else None
            cand_q, cand_t, rejections = [], [], {}
            for c in range(C):
                try:
                    q = self.candidate_joint_q(actions[c])
                except ProposalRejected as error:
                    q = self.binding.joint_q.copy(); rejections[c] = str(error)
                t = self.baseline.joint_target.copy()
                t[:, self.projector.dids] = q[:, self.projector.qids]
                for qi, di in self.projector.ball_joints:
                    t[:, di:di+3] = Rotation.from_quat(q[:, qi:qi+4]).as_rotvec()
                cand_q.append(q); cand_t.append(t)
            ilc_info = {}
            if self.options.get('ilc_candidates'):
                for c, (info, q, tt) in enumerate(self._ilc_candidates(learner)[:C]):
                    cand_q[c], cand_t[c] = q, tt; rejections.pop(c, None); ilc_info[c] = info
                print(f'[ilc] step={step} {len(ilc_info)} physics-feedback candidates; {self._ilc_feedback}', flush=True)
            if win_mask is not None:
                for c in range(C):
                    cand_q[c], cand_t[c] = self._window_blend(cand_q[c], win_mask)
            bq, bt = self.binding.joint_q.copy(), self.binding.joint_target.copy()
            qs = np.asarray([q for q in cand_q for _ in range(R)] + [bq] * NB)
            targets = np.asarray([t for t in cand_t for _ in range(R)] + [bt] * NB)
            world_actions = np.concatenate([np.repeat(actions[:C], R, axis=0), np.zeros((NB, actions.shape[1]), actions.dtype)])
            if win_mask is not None:
                _co = world_actions[:, :wd].reshape(len(world_actions), self.basis.shape[1], self.joint_basis.shape[0]).copy()
                _co[:, ~self._win_chunks] = 0; world_actions[:, :wd] = _co.reshape(len(world_actions), -1)
            for c in ilc_info: world_actions[c*R:(c+1)*R] = 0  # not samples of the upper policy
            print(f'[parallel] step={step} evaluating {C} candidates x {R} replicas + {NB} baseline replicas', flush=True)
            metrics, packed, _ = rollout_candidates(self, learner, qs, targets)
            costs = np.asarray([cost(m, q) for m, q in zip(metrics, qs)])
            costs = np.where(np.isfinite(costs), costs, 50.)
            group = costs[:C*R].reshape(C, R); base_costs = costs[C*R:]
            cmean, cse = group.mean(1), group.std(1, ddof=1) / np.sqrt(R)
            bmean, bse = float(base_costs.mean()), float(base_costs.std(ddof=1) / np.sqrt(max(NB, 2)))
            def agg(rows):
                return {k: float(np.mean([m[k] for m in rows])) for k in ('object_position_robust_m', 'object_orientation_robust_rad',
                        'object_success_robust_fraction', 'unstable_world_fraction', 'contact_proximity_mean_m') if k in rows[0]}
            base_agg = agg(metrics[C*R:]); cand_agg = [agg(metrics[c*R:(c+1)*R]) for c in range(C)]
            self._last_base_rows = list(metrics[C*R:])
            finite = [all(m['all_errors_finite'] for m in metrics[c*R:(c+1)*R]) for c in range(C)]
            margin = 2 * np.sqrt(cse**2 + bse**2)
            valid = np.asarray([c not in rejections and finite[c] and (bmean - cmean[c]) > margin[c]
                and cand_agg[c]['object_position_robust_m'] <= base_agg['object_position_robust_m'] + .002
                and cand_agg[c]['object_orientation_robust_rad'] <= base_agg['object_orientation_robust_rad'] + .01
                and cand_agg[c]['object_success_robust_fraction'] >= base_agg['object_success_robust_fraction'] - .01
                for c in range(C)])
            rewards = np.concatenate([np.repeat(np.clip(bmean - cmean, -5., 5.), R), np.zeros(NB)])
            for c in rejections: rewards[c*R:(c+1)*R] = -5.
            for c in ilc_info: rewards[c*R:(c+1)*R] = 0.
            action = jnp.asarray(world_actions, dtype=action.dtype)
            transition = Transition(action=action, objective_terms=jnp.asarray(rewards[:,None], jnp.float32),
                terminated=jnp.ones(W,jnp.int32), truncated=jnp.zeros(W,jnp.int32),
                episode_reference_progress=jnp.ones(W,jnp.float32), termination_diagnostics=jnp.zeros((W,0),jnp.float32),
                next_observation=obs,next_critic_context=context,post_reset_observation=obs,post_reset_critic_context=context)
            self.upper, _, losses = self.executables.consume_transition(self.upper,transition,
                jnp.ones(1,jnp.float32),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
            jax.block_until_ready(self.upper)
            selected = int(np.argmin(np.where(valid, cmean, np.inf))) if valid.any() else None
            accepted = selected is not None
            self._proposal_pool = (cand_q, cand_t)
            record = dict(step=step, engine='parallel_flashsac', upper_round=self.upper_round, version_before=self.version,
                baseline=baseline, lower_policy_frozen=True, candidate_worlds=W, replicas_per_candidate=R, baseline_replicas=NB,
                object_contact_sha256=self.command_sha256,
                replica_baseline=dict(cost_mean=bmean, cost_se=bse, **base_agg),
                candidates=[dict(trial=c, cost_mean=float(cmean[c]), cost_se=float(cse[c]), gate_passed=bool(valid[c]),
                                 rejected=rejections.get(c), ilc=ilc_info.get(c), **cand_agg[c]) for c in range(C)])
            if win_mask is not None:
                record['failure_window'] = self._win_info
            if ilc_info:
                record['ilc_feedback'] = self._ilc_feedback
                print(f'[ilc] step={step} selected={selected} ' + ('ILC ' + str(ilc_info[selected]) if selected in ilc_info else 'random/none'), flush=True)
            evidence = self.output.parent/'search_batches'/f'step_{step}'
            evidence.mkdir(parents=True,exist_ok=True)
            _top = np.argsort(cmean)[:16]
            np.savez_compressed(evidence/'proposals.npz', actions=actions[:C], cost_mean=cmean, cost_se=cse, valid=valid,
                                baseline_cost_mean=bmean, baseline_cost_se=bse, top_trials=_top,
                                top_joint_q=np.asarray(cand_q)[_top], top_joint_target=np.asarray(cand_t)[_top])
            if accepted:
                candidate = build_robot_reference_binding(self.projector.builder,self.projector.layout,
                    cand_q[selected],cand_t[selected],self.baseline.fps)
                save_snapshot(evidence/'selected_reference.npz',candidate,self.setup.scene.layout,
                              provenance=dict(step=step,trial=selected,accepted=True,object_contact_sha256=self.command_sha256))
                record['validation'] = dict(replica_mean_cost=float(cmean[selected]), replica_se=float(cse[selected]),
                                            baseline_mean_cost=bmean, baseline_se=bse, candidate=cand_agg[selected], baseline=base_agg)
                _space = getattr(self, 'grasp_proto', None) or getattr(self, 'oc', None)
                if _space is not None:
                    record['selected_grasp_prototypes'] = _space.describe(actions[selected][wd:])
                self.binding=candidate;self.version+=1;self.publish(step)
                if getattr(self, 'oc', None) is not None:
                    self.oc.commit(actions[selected][wd:])
                    record['committed_grasps'] = self.oc.describe(np.zeros(self.oc.action_dim, np.float32))
        else:
            qs, targets, rejections = [], [], {}
            for world, proposal in enumerate(actions):
                try:
                    q = self.candidate_joint_q(proposal)
                except ProposalRejected as error:
                    q = self.binding.joint_q.copy()
                    rejections[world] = str(error)
                target = self.baseline.joint_target.copy()
                target[:, self.projector.dids] = q[:, self.projector.qids]
                for qi, di in self.projector.ball_joints:
                    target[:, di:di+3] = Rotation.from_quat(q[:, qi:qi+4]).as_rotvec()
                qs.append(q); targets.append(target)
            qs, targets = np.asarray(qs), np.asarray(targets)
            print(f'[parallel] step={step} evaluating {self.upper_worlds} distinct candidates together', flush=True)
            metrics, packed, _ = rollout_candidates(self, learner, qs, targets)
            costs = np.asarray([cost(m, q) for m, q in zip(metrics, qs)])
            valid = np.asarray([i not in rejections and gate(m, baseline) and np.isfinite(c)
                                for i, (m, c) in enumerate(zip(metrics, costs))])
            rewards = np.clip(np.nan_to_num(baseline_cost-costs, nan=-5., posinf=-5., neginf=-5.), -5., 5.)
            for i in rejections: rewards[i] = -5.
            transition = Transition(action=action, objective_terms=jnp.asarray(rewards[:,None], jnp.float32),
                terminated=jnp.ones(self.upper_worlds,jnp.int32), truncated=jnp.zeros(self.upper_worlds,jnp.int32),
                episode_reference_progress=jnp.ones(self.upper_worlds,jnp.float32),
                termination_diagnostics=jnp.zeros((self.upper_worlds,0),jnp.float32),
                next_observation=obs,next_critic_context=context,post_reset_observation=obs,post_reset_critic_context=context)
            self.upper, _, losses = self.executables.consume_transition(self.upper,transition,
                jnp.ones(1,jnp.float32),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
            jax.block_until_ready(self.upper)
            selected = int(np.argmin(np.where(valid, costs, np.inf))) if valid.any() else None
            accepted = False
            record = dict(step=step, engine='parallel_flashsac', upper_round=self.upper_round,
                version_before=self.version, baseline=baseline, lower_policy_frozen=True,
                candidate_worlds=self.upper_worlds, replicas_per_candidate=1,
                object_contact_sha256=self.command_sha256,
                candidates=[dict(trial=i,cost=float(c),reward=float(r),metrics=m,gate_passed=bool(v),
                                 rejected=rejections.get(i)) for i,(c,r,m,v) in enumerate(zip(costs,rewards,metrics,valid))])
            evidence = self.output.parent/'search_batches'/f'step_{step}'
            evidence.mkdir(parents=True,exist_ok=True)
            # Preserve all proposals for reproducible later comparisons, not only the winner.
            # Slim evidence: every candidate's action + cost (trajectories are reproducible from them),
            # full trajectories only for the 16 lowest-cost candidates. Full dumps cost ~20 s and 250 MB/round.
            _top=np.argsort(np.where(np.isfinite(costs),costs,np.inf))[:16]
            np.savez_compressed(evidence/'proposals.npz',actions=actions,costs=costs,valid=valid,top_trials=_top,
                                top_joint_q=qs[_top],top_joint_target=targets[_top])
            if selected is not None and costs[selected] < baseline_cost:
                candidate = build_robot_reference_binding(self.projector.builder,self.projector.layout,
                    qs[selected],targets[selected],self.baseline.fps)
                check, _, _ = self.rollout(candidate,learner,seed)
                accepted = gate(check,baseline) and cost(check,candidate.joint_q) < baseline_cost
                save_snapshot(evidence/'selected_reference.npz',candidate,self.setup.scene.layout,
                              provenance=dict(step=step,trial=selected,accepted=bool(accepted),object_contact_sha256=self.command_sha256))
                _space = getattr(self, 'grasp_proto', None) or getattr(self, 'oc', None)
                if _space is not None:
                    record['selected_grasp_prototypes'] = _space.describe(actions[selected][self.basis.shape[1]*self.joint_basis.shape[0]:])
                record['validation'] = dict(baseline=baseline,candidate=check,
                    proposal_source=f'upper_trial_{selected}',deterministic_confirmation=True)
                if accepted:
                    self.binding=candidate;self.version+=1;self.publish(step)
                    if getattr(self, 'oc', None) is not None:
                        # Accepted object-centric grasps persist: alpha<=0 later means "keep".
                        self.oc.commit(actions[selected][self.basis.shape[1]*self.joint_basis.shape[0]:])
                        record['committed_grasps'] = self.oc.describe(np.zeros(self.oc.action_dim, np.float32))
        save_learner_checkpoint(self.output.parent/'upper',self.upper,
            metadata={'lower_step':step,'engine':'parallel_flashsac'},save_replay=True)
        if command_digest(self.setup.reference)!=self.command_sha256:
            raise RuntimeError('upper editor mutated immutable object/contact commands')
        record.update(accepted=bool(accepted),version_after=self.version,selected_trial=selected,
            upper_optimizer_updates=int(self.upper.global_update_step),seconds=time.perf_counter()-started,
            upper_simulation_transitions=self.validation_steps-previous_steps,
            cumulative_upper_simulation_transitions=self.validation_steps)
        # Reject nonfinite diagnostic records instead of publishing corrupt JSON.
        self._last_record = record
        if record.get('accepted'):
            self._accepted_stats = dict(((record.get('validation') or {}).get('candidate') or {}))
        payload=json.dumps(_json_safe(record),allow_nan=False)+'\n'
        (evidence/'scores.json').write_text(payload)
        with (self.output.parent/'upper_updates.jsonl').open('a') as stream:stream.write(payload)
        print(f'[parallel] step={step} accepted={accepted} seconds={record["seconds"]:.1f}',flush=True)
        try:
            with (self.output.parent/'upper_timing.jsonl').open('a') as _s:_s.write(json.dumps({'step':step,'round_seconds':round(record['seconds'],2)})+'\n')
        except Exception:pass
        if self.options.get('train_on_proposals') and getattr(self, '_proposal_pool', None) is not None:
            return self._proposal_training_env(env)
        if not accepted:return None  # keep replay/observations when the reference is unchanged
        replacement=RLEnv(replace(self.setup.scene,robot_reference=self.binding),self.setup.reference,config=deepcopy(env.env.config))
        assert_device_binding(replacement,self.binding)
        np.testing.assert_array_equal(replacement.command.reference.body_pos_w.numpy(),env.env.command.reference.body_pos_w.numpy())
        np.testing.assert_array_equal(replacement.command.reference.body_quat_w.numpy(),env.env.command.reference.body_quat_w.numpy())
        replacement.reset();replacement.capture_step();replacement.reset()
        return WarpRLEnv(replacement,publish_diagnostics=False,publish_step_metrics=False)


def _json_safe(x):
    """recursively replace non-finite floats (exploded worlds) with None so the record can always be written"""
    import math
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, 'item') and not isinstance(x, (str, bytes)):
        try:
            return _json_safe(x.item())
        except Exception:
            return x
    return x
