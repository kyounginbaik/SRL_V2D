"""One contact-focused full-hand or wrist reference increment per search round."""
import json
import numpy as np
from flash_chord.retargeting.parallel_editor import ParallelReferenceEditor
from flash_chord.retargeting.local_correction import local_candidate, local_envelope
from flash_chord.retargeting.ttsa_editor import upper_cost, passes_study_gate, contact_proximity_gate


def scheduled_contact_context(schedule, round_index, frames, fps, feedback_shape):
    """Resolve an episode-specific window and explicit hand/object target."""
    item = schedule[(round_index - 1) % len(schedule)]
    start = float(item['start_seconds'])
    hand, body = int(item['hand_index']), int(item['object_index'])
    duration = (frames - 1) / fps
    if not np.isfinite(start) or not 0 <= start < duration:
        raise ValueError('contact schedule window is outside the episode')
    if not (0 <= hand < feedback_shape[1] and 0 <= body < feedback_shape[2]):
        raise ValueError('contact schedule target is outside the hand/object roster')
    return dict(bin=min(int(start), feedback_shape[0]-1), start_seconds=start,
                end_seconds=min(start+1., duration), hand_index=hand, object_index=body,
                focus=str(item['focus']))


def confirmation_contact_gate(editor, candidate, baseline):
    tolerance=float(editor.options.get('contact_confirmation_tolerance_m',.002))
    if not np.isfinite(tolerance) or not 0 <= tolerance <= .01:
        raise ValueError('contact confirmation tolerance must be between 0 and 1 cm')
    if not contact_proximity_gate(candidate,baseline,per_object_tolerance_m=tolerance):return False
    context=getattr(editor,'local_context',None)
    if context:
        if 'hand_index' in context and 'object_index' in context:
            hand,body=int(context['hand_index']),int(context['object_index'])
        else:
            hand,body=(0,0) if context['focus']=='left_pot' else (1,1)
        before=baseline['per_hand_object_contact_proximity_mean_m'][hand][body]
        after=candidate['per_hand_object_contact_proximity_mean_m'][hand][body]
        if before is None or after is None or after > before-.001:return False
    return True


class LocalReferenceEditor(ParallelReferenceEditor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dimensions = 12 if self.options.get('edit_scope') == 'wrists_only' else 56
        if self.joint_basis.shape[0] != dimensions:
            raise ValueError('local editor requires floating bimanual controls matching its edit scope')
        self.visits = np.zeros(self.basis.shape[1], int)
        self.local_bin = 6
        self.basis = local_envelope(self.baseline.num_frames, self.baseline.fps, 6., 7.)[:, None]
        p = self.output.parent/'ttsa_manifest.json'
        manifest = json.loads(p.read_text())
        manifest.update(upper_action_dimension=dimensions, temporal_modes=1,
            proposal_projection='local_increment_cumulative_bounds', publication_blend=1.,
            local_shoulder_seconds=.5, upper_updates_per_batch=self.options.get('upper_updates', 2))
        p.write_text(json.dumps(manifest, indent=2)+'\n')

    def candidate_joint_q(self, action):
        return local_candidate(self, action)

    def observation(self, metrics, feedback=None):
        # Alternate the troublesome pot interval and the worst active lid interval.
        vectors = np.asarray(feedback['contact_direction_bins'])
        counts = np.asarray(feedback['contact_direction_counts'])
        errors = np.linalg.norm(vectors, axis=-1)
        contact_schedule = self.options.get('local_contact_schedule')
        if contact_schedule:
            context = scheduled_contact_context(contact_schedule, self.upper_round,
                self.baseline.num_frames, self.baseline.fps, errors.shape)
            hand, body = context['hand_index'], context['object_index']
            self.local_bin = context['bin']
            self.visits[self.local_bin] += 1
            self.basis = local_envelope(self.baseline.num_frames, self.baseline.fps,
                context['start_seconds'], context['end_seconds'])[:, None]
            context['contact_error_m'] = float(errors[self.local_bin, hand, body])
            self.local_context = context
            obs = super().observation(metrics, feedback)
            import jax.numpy as jnp
            duration = (self.baseline.num_frames-1)/self.baseline.fps
            return jnp.concatenate([obs, jnp.asarray([[context['start_seconds']/duration,
                float(hand == 0)]], dtype=jnp.float32)], axis=1)
        pot_turn = self.upper_round % 2 == 1
        active = errors[:, 0, 0] > 0 if pot_turn else errors[:, 1, 1] > 0
        score = errors[:, 0, 0] if pot_turn else errors[:, 1, 1]
        allowed = active.copy()
        if pot_turn:
            allowed &= (np.arange(len(active)) >= 5) & (np.arange(len(active)) < 10)
        if not allowed.any(): allowed = active
        if not allowed.any(): allowed = np.ones(len(active), bool)
        # Repeat failing intervals, but prevent one interval monopolizing all rounds.
        priority = np.where(allowed, score/(1+.25*self.visits), -np.inf)
        schedule = self.options.get('local_segment_schedule')
        self.local_bin = int(schedule[(self.upper_round-1)%len(schedule)]) if schedule else int(np.argmax(priority))
        self.visits[self.local_bin] += 1
        start = self.local_bin*float(self.options.get('chunk_seconds', 1.))
        self.basis = local_envelope(self.baseline.num_frames, self.baseline.fps,
            start, min(start+1., (self.baseline.num_frames-1)/self.baseline.fps))[:, None]
        self.local_context = dict(bin=self.local_bin, start_seconds=start, end_seconds=start+1.,
            focus='left_pot' if pot_turn else 'right_lid', contact_error_m=float(score[self.local_bin]))
        obs = super().observation(metrics, feedback)
        import jax.numpy as jnp
        return jnp.concatenate([obs, jnp.asarray([[start/17., float(pot_turn)]], dtype=jnp.float32)], axis=1)

    def proposal_diagnostics(self, actions, qs):
        ids = [q for n,q in zip(self.projector.layout.scalar_joints.names,self.projector.qids)
               if n.endswith(('wrist_x','wrist_y','wrist_z'))]
        difference = qs[:,:,ids]-self.binding.joint_q[None,:,ids]
        local = self.basis[:,0] > 0
        requested=(actions @ self.joint_basis)[:,None,ids] * self.basis[None,:,:] * float(self.options.get('local_translation_step_m',.02))
        return dict(local_context=self.local_context,
            requested_translation_mean_abs_m=float(np.mean(abs(requested[:,local]))),
            proposal_translation_mean_abs_m=float(np.mean(abs(difference[:,local]))),
            proposal_translation_max_abs_m=float(np.max(abs(difference))),
            outside_segment_change_max=float(np.max(abs(qs[:,~local]-self.binding.joint_q[None,~local]))),
            upper_updates_per_batch=int(self.options.get('upper_updates',2)))

    def confirm_candidate(self, candidate, learner, seed, baseline, nominal):
        if not self.options.get('robust_confirmation', True):
            return True, {'enabled':False}
        checks = []
        for test in range(3):
            rng = np.random.default_rng(9101+test)
            delta = np.zeros(self.baseline.num_joint_q)
            for name, qid in zip(self.projector.layout.scalar_joints.names, self.projector.qids):
                delta[qid] = rng.uniform(-.001,.001) if name.endswith(('wrist_x','wrist_y','wrist_z')) else rng.uniform(-np.deg2rad(.5),np.deg2rad(.5))
            base,_,_ = self.rollout(self.binding,learner,seed,initial_q_delta=delta)
            after,_,_ = self.rollout(candidate,learner,seed,initial_q_delta=delta)
            def cost(m,q):
                return upper_cost(m,q-self.baseline.joint_q,self.baseline.fps,1.,robust=True,
                    contact_proximity_weight=self.contact_proximity_weight)
            passed = passes_study_gate(after,base) and confirmation_contact_gate(self,after,base)
            checks.append(dict(seed=9101+test,baseline=base,candidate=after,gate_passed=bool(passed),
                cost_gain=cost(base,self.binding.joint_q)-cost(after,candidate.joint_q)))
        def average(values):
            if values[0] is None:
                if not all(v is None for v in values): raise ValueError('contact masks differ across paired resets')
                return None
            if isinstance(values[0], bool): return all(values)
            if isinstance(values[0], list): return [average([v[i] for v in values]) for i in range(len(values[0]))]
            return float(np.mean(values))
        mean_before = {k:average([c['baseline'][k] for c in checks]) for k in checks[0]['baseline']}
        mean_after = {k:average([c['candidate'][k] for c in checks]) for k in checks[0]['candidate']}
        ok = (mean_after.get('all_errors_finite',False) and mean_after.get('full_horizon',False)
            and mean_before.get('all_errors_finite',False) and mean_before.get('full_horizon',False)
            and sum(c['cost_gain']>0 for c in checks)>=2 and np.mean([c['cost_gain'] for c in checks])>0
            and passes_study_gate(mean_after,mean_before) and confirmation_contact_gate(self,mean_after,mean_before))
        return bool(ok),dict(enabled=True,passed=bool(ok),checks=checks,mean_baseline=mean_before,mean_candidate=mean_after,
            wrist_translation_perturbation_m=.001,finger_perturbation_deg=.5)
