"""Jointly search local full-hand corrections and failure-informed locations."""
import json
import numpy as np
import jax.numpy as jnp
from flash_chord.retargeting.local_editor import LocalReferenceEditor
from flash_chord.retargeting.local_correction import local_envelope
from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor, upper_cost, passes_study_gate, contact_proximity_gate
from flash_chord.retargeting.failure_windows import failure_timing, propose_windows, trajectory_metrics, prefix_gate, object_pose_cost


def mean_metrics(rows):
    def average(values):
        if values[0] is None:
            if not all(v is None for v in values):
                raise ValueError('paired contact masks differ')
            return None
        if isinstance(values[0], bool):
            return all(values)
        if isinstance(values[0], list):
            return [average([v[i] for v in values]) for i in range(len(values[0]))]
        return float(np.mean(values))
    return {k: average([row[k] for row in rows]) for k in rows[0]}


class AdaptiveWindowEditor(LocalReferenceEditor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.options.get('local_segment_schedule') or self.options.get('local_contact_schedule'):
            raise ValueError('adaptive selection must not receive a fixed window schedule')
        self.window_visits = {}
        self.window_rng = np.random.default_rng(self.training.seed+31000)
        self.protected_prefix_frames = 0
        self.window_contexts = []
        self.timing_options = dict(
            position_limit=float(self.options.get('failure_position_m', .05)),
            angle_limit=np.deg2rad(float(self.options.get('failure_orientation_deg', 20.))),
            persistence_seconds=float(self.options.get('failure_persistence_seconds', .15)))
        manifest_path = self.output.parent/'ttsa_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(engine='adaptive_window_flashsac', window_selection='failure_onset_and_sampled_history',
                        max_windows=4, upper_action_dimension=int(self.joint_basis.shape[0]), window_context_in_upper_observation=True,
                        nominal_candidates_per_window=self.upper_worlds//4,
                        confirmation='three paired perturbed resets, both objects and successful prefix',
                        contact_target='no preassigned hand or object', search_voc_scale=0.)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    def trajectory_metrics(self, p, a):
        return trajectory_metrics(p, a, self.baseline.fps, self.protected_prefix_frames, **self.timing_options)

    def observation(self, metrics, feedback=None):
        position = np.asarray(feedback['frame_per_object_position_m']).mean(axis=1)
        orientation = np.asarray(feedback['frame_per_object_orientation_rad']).mean(axis=1)
        if 'baseline_ensemble_position_m' in feedback:
            from flash_chord.retargeting.failure_consensus import failure_consensus
            ensemble_p = np.asarray(feedback['baseline_ensemble_position_m'])
            ensemble_a = np.asarray(feedback['baseline_ensemble_orientation_rad'])
            self.failure_context = failure_consensus(ensemble_p, ensemble_a, self.baseline.fps,
                minimum_fraction=float(self.options.get('failure_consensus_fraction', .5)),
                cluster_seconds=float(self.options.get('failure_consensus_cluster_seconds', .5)),
                **self.timing_options)
            self.failure_context.update(feedback['baseline_ensemble_metadata'])
            # Context summarizes the unchanged-reference population; timing is
            # computed BEFORE aggregation so large outliers cannot move the trigger.
            position, orientation = np.median(ensemble_p, axis=1), np.median(ensemble_a, axis=1)
        else:
            self.failure_context = failure_timing(position, orientation, self.baseline.fps, **self.timing_options)
        self.protected_prefix_frames = self.failure_context['onset_frame'] if self.failure_context['failed'] else len(position)
        # The first baseline rollout preceded window selection. Refresh its
        # prefix metrics so baseline and every candidate use identical frames.
        metrics.update(self.trajectory_metrics(position, orientation))
        self.window_contexts = propose_windows(self.failure_context, self.baseline.num_frames,
            self.baseline.fps, self.window_rng, self.window_visits)
        self.window_ids = np.empty(self.upper_worlds, int)
        for w, ids in zip(self.window_contexts, np.array_split(np.arange(self.upper_worlds), len(self.window_contexts))):
            self.window_ids[ids] = w['window_id']
            w['candidate_count'] = len(ids)
        self.window_envelopes = np.stack([local_envelope(self.baseline.num_frames, self.baseline.fps,
            w['start_seconds'], w['end_seconds']) for w in self.window_contexts])
        self.before_search_q = self.binding.joint_q.copy()
        base = np.asarray(TTSAReferenceEditor.observation(self, metrics, feedback))[0]
        duration = (self.baseline.num_frames-1)/self.baseline.fps
        rows = []
        for w in self.window_contexts:
            lo = int(w['start_seconds']*self.baseline.fps)
            hi = max(lo+1, int(w['end_seconds']*self.baseline.fps))
            context = [w['start_seconds']/duration, w['end_seconds']/duration,
                       (self.failure_context['failure_seconds'] or 0.)/duration,
                       self.failure_context['onset_seconds']/duration, float(self.failure_context['failed'])]
            rows.append(np.r_[base, context, position[lo:hi].mean(0)/.05, orientation[lo:hi].mean(0)/np.deg2rad(20)])
        self.select_candidate_context(0)
        print('[adaptive] '+json.dumps(dict(failure=self.failure_context, windows=self.window_contexts)), flush=True)
        return jnp.asarray(np.asarray(rows)[self.window_ids], dtype=jnp.float32)

    def select_candidate_context(self, trial):
        index = self.window_ids[trial]
        self.basis = self.window_envelopes[index, :, None]
        self.local_context = dict(self.window_contexts[index])

    def proposal_diagnostics(self, actions, qs):
        outside_max = 0.
        for i, q in enumerate(qs):
            outside = self.window_envelopes[self.window_ids[i]] == 0
            outside_max = max(outside_max, float(np.max(abs(q[outside]-self.before_search_q[outside]))))
        return dict(failure_context=self.failure_context, windows=self.window_contexts,
                    outside_segment_change_max=outside_max, upper_updates_per_batch=int(self.options.get('upper_updates', 32)))

    def acceptance_gate(self, after, before):
        # Object tracking must improve separately in confirmation. Contact can
        # remain unchanged: a collision fix need not improve a chosen grasp.
        return (after['all_errors_finite'] and after['full_horizon'] and
                passes_study_gate(after, before) and prefix_gate(after, before) and
                contact_proximity_gate(after, before, minimum_gain_m=-.001,
                    per_object_tolerance_m=float(self.options.get('contact_confirmation_tolerance_m', .01))))

    def shortlist_candidates(self, ranked, eligible, costs, baseline_cost):
        """Give each promising window a confirmation opportunity before repeats."""
        candidates = [int(i) for i in ranked if eligible[i] and costs[i] < baseline_cost]
        selected, seen = [], set()
        for i in candidates:
            window = int(self.window_ids[i])
            if window not in seen:
                selected.append(i)
                seen.add(window)
        limit = int(self.options.get('confirmation_top_k', 4))
        selected += [i for i in candidates if i not in selected][:max(0, limit-len(selected))]
        return selected[:limit]

    def confirm_candidate(self, candidate, learner, seed, baseline, nominal):
        checks = []
        for test in range(3):
            rng = np.random.default_rng(9101+test)
            delta = np.zeros(self.baseline.num_joint_q)
            for name, qid in zip(self.projector.layout.scalar_joints.names, self.projector.qids):
                delta[qid] = rng.uniform(-.001, .001) if name.endswith(('wrist_x','wrist_y','wrist_z')) else rng.uniform(-np.deg2rad(.5), np.deg2rad(.5))
            before, _, _ = self.rollout(self.binding, learner, seed, initial_q_delta=delta)
            after, _, _ = self.rollout(candidate, learner, seed, initial_q_delta=delta)
            def cost(m, q):
                return upper_cost(m, q-self.baseline.joint_q, self.baseline.fps, 1., robust=True,
                                  contact_proximity_weight=self.contact_proximity_weight)
            checks.append(dict(seed=9101+test, baseline=before, candidate=after,
                gate_passed=bool(self.acceptance_gate(after, before)),
                cost_gain=cost(before, self.binding.joint_q)-cost(after, candidate.joint_q),
                object_pose_cost_gain=object_pose_cost(before)-object_pose_cost(after)))
        before = mean_metrics([c['baseline'] for c in checks])
        after = mean_metrics([c['candidate'] for c in checks])
        ok = (self.acceptance_gate(after, before) and before['all_errors_finite'] and before['full_horizon']
              and sum(c['cost_gain'] > 0 for c in checks) >= 2
              and np.mean([c['cost_gain'] for c in checks]) > 0
              and np.mean([c['object_pose_cost_gain'] for c in checks]) > 1e-4
              and sum(c['object_pose_cost_gain'] > 0 for c in checks) >= 2)
        return bool(ok), dict(enabled=True, passed=bool(ok), checks=checks, mean_baseline=before,
                             mean_candidate=after, wrist_translation_perturbation_m=.001,
                             finger_perturbation_deg=.5, protected_prefix_seconds=self.protected_prefix_frames/self.baseline.fps)

    def finalize_search_record(self, record, qs, costs, valid):
        selected = record['selected_trial']
        record['selected_window'] = (self.window_contexts[self.window_ids[selected]]
                                     if record['accepted'] and selected is not None else None)
        record['window_results'] = [dict(w, best_cost=float(costs[self.window_ids == w['window_id']].min()),
            improving_count=int(np.sum(costs[self.window_ids == w['window_id']] < record['gate_diagnostics']['baseline_cost'])),
            screening_pass_count=int(np.sum(valid[self.window_ids == w['window_id']]))) for w in self.window_contexts]
        change = {}
        for hand in self.projector.layout.hands:
            side = hand.side
            ids = [qid for name, qid in zip(self.projector.layout.scalar_joints.names, self.projector.qids)
                   if name.startswith(side+'_') and name.endswith(('wrist_x','wrist_y','wrist_z'))]
            # Use layout-provided fingers; wrist names are validated by the 56D embodiment.
            for label, origin in [('accepted_increment', self.before_search_q), ('cumulative', self.baseline.joint_q)]:
                d = self.binding.joint_q-origin
                change[f'{label}_{side}_wrist_peak_cm'] = float(np.linalg.norm(d[:, ids], axis=1).max()*100)
                f = d[:, hand.finger_q_ids]
                change[f'{label}_{side}_finger_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean(f*f))))
                change[f'{label}_{side}_finger_peak_deg'] = float(np.rad2deg(abs(f).max()))
        record['reference_change'] = change
        path = self.output.parent/'adaptive_search_summary.jsonl'
        keys = ('step','upper_round','accepted','version_after','seconds','failure_context','windows',
                'selected_window','window_results','reference_change','gate_diagnostics')
        summary = {k:record[k] for k in keys}
        robust = record.get('robust_confirmation', {})
        if robust.get('checks'):
            summary['paired_cost_gain_mean'] = float(np.mean([c['cost_gain'] for c in robust['checks']]))
            summary['paired_object_pose_gain_mean'] = float(np.mean([c['object_pose_cost_gain'] for c in robust['checks']]))
            summary['paired_baseline'] = robust['mean_baseline']
            summary['paired_candidate'] = robust['mean_candidate']
        with path.open('a') as stream:
            stream.write(json.dumps(summary, allow_nan=False)+'\n')
        (self.output.parent/'window_visits.json').write_text(json.dumps(self.window_visits, indent=2)+'\n')
        print('[adaptive-result] '+json.dumps({k:summary[k] for k in ('step','accepted','selected_window','reference_change')}), flush=True)
