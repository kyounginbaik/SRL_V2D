"""Online replay -> wrench critic -> unrolled lower actor -> upper actor.

Opt-in engine. The simulator remains the acceptance oracle; neither learned
cost predictions nor a nonzero hypergradient alone authorize publication.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from flash_chord.retargeting.response_editor import ResponseBilevelEditor
from flash_chord.retargeting.reference_relabel import ReferenceRelabel, binding_for_q
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.retargeting.wrench_hypergradient import (
    HypergradientConfig, ReferenceReplay, WrenchHypergradient, TrialRollout, actor_policy_digest,
)
from flash_chord.runtime.jax import to_jax
from flash_chord.training.reference_update import ReferenceLearnerUpdate
from flash_chord.retargeting.lower_adaptation import (
    AdaptationConfig, FlashSACAdaptation, learning_snapshot, install_learning,
    training_readiness, comparison_stable, comparison_instability_reasons,
)


def support_cost(current, reference, *, tolerance=.1):
    """Per-world fixed-target support loss; equal object weighting, all hands."""
    if current.shape != reference.shape or current.ndim != 4:
        raise ValueError('support arrays must be matching [world, hand, object, basis]')
    active = reference > 1e-3
    cell = active.any(axis=-1)
    under = np.maximum((1-tolerance)*reference-current, 0)
    over = np.maximum(current-(1+tolerance)*reference, 0)
    band = np.minimum(np.mean(under**2+over**2, axis=-1) /
                      np.maximum(np.mean(reference**2, axis=-1), 1e-6), 10.)
    missed = np.sum(active & (current <= 1e-3), axis=-1)/np.maximum(active.sum(axis=-1), 1)
    wanted = np.sum(np.where(cell, 5*band+missed, 0.), axis=1)/np.maximum(cell.sum(axis=1), 1)
    unwanted = np.sum(np.where(~cell, np.minimum(np.mean(current**2, axis=-1), 10.), 0.), axis=1) / \
        np.maximum((~cell).sum(axis=1), 1)
    return np.mean(wanted+unwanted, axis=-1).astype(np.float32)


class WrenchHypergradientEditor(ResponseBilevelEditor):
    def __init__(self, **kwargs):
        options = dict(kwargs['options'])
        if options.get('upper_proposal', 'gaussian') not in ('gaussian', 'hypergradient'):
            raise ValueError('this engine uses its own hypergradient actor, not the bandit FlashSAC actor')
        kwargs['options'] = {**options, 'upper_proposal': 'gaussian'}
        super().__init__(**kwargs)
        if not self.fixed_start or not self.local_acceptance:
            raise ValueError('wrench hypergradient requires fixed_start and local_nominal acceptance')
        self.hg_config = HypergradientConfig(**dict(options.get('hypergradient', {})))
        self.adaptation_config = AdaptationConfig(**dict(options.get('adaptation', {})))
        self.adaptation_cache = None
        self.adaptation_transitions = 0
        self.replay = ReferenceReplay(int(options.get('hyper_replay_capacity', 32768)))
        if self.replay.capacity < max(2*self.hg_config.batch_size, 256):
            raise ValueError('hyper replay capacity cannot reach the minimum batch requirement')
        self.replay.clear(self.version)
        self.capture_stride = int(options.get('hyper_capture_stride', 4))
        self.capture_worlds = int(options.get('hyper_capture_worlds', 32))
        if min(self.capture_stride, self.capture_worlds) < 1:
            raise ValueError('positive capture stride/world count required')
        self.pending = None
        self.capture_evaluation = False
        self.trial_capture = None
        self.baseline_capture = None
        self._round_critic_feedback = []
        self.hyper = None
        self.hyper_rng = np.random.default_rng(self.training.seed+93001)
        self._metrics = {}
        path = self.output.parent/'ttsa_manifest.json'
        manifest = json.loads(path.read_text())
        manifest.update(engine='wrench_hypergradient', upper_learning='actor_with_unrolled_SAC_actor_response',
            hypergradient=asdict(self.hg_config), upper_action_dimension=56,
            adaptation=asdict(self.adaptation_config),
            upper_trials='one learned proposal per detected window, paired with adapted no-edit',
            guarantee_scope='autodiff of finite projected actor-optimizer response and learned cost surrogate; '
                            'no full-RL optimum or true-physics gradient guarantee',
            proposal_response=f'{self.hg_config.inner_steps} actor {self.hg_config.inner_optimizer} updates; '
                              'Adam preserves incoming moments/schedule; critic, statistics and replay states frozen',
            lower_response='matched full FlashSAC adaptation on fresh simulated experience before acceptance',
            reference_derivative='feasible finite differences and native FK; one-sided at active limits; fixed-state relabeling',
            proposal_lower_objective='SAC actor loss under relabeled fixed replay and frozen lower critic',
            lower_objective='ordinary configured FlashSAC actor, critic and temperature losses on fresh data',
            lower_replay='cleared on accepted reference change; retained after rejection',
            upper_replay='reference-version-isolated transitions; current-round-only outer baseline; isolated policy-matched trial feedback',
            trial_feedback='each tested control/candidate checkpoint teaches critic with measured TD and complete-rollout returns, including rejected proposals',
            upper_updates_per_batch=self.hg_config.upper_steps,
            nominal_candidates_per_window=1, candidate_worlds=1, eval_worlds=1,
            upper_feedback='discounted CHORD-style wrench-support cost, not measured six-axis net wrench',
            upper_cost_scale=self.hg_config.cost_scale,
            early_termination='omit early tracking cutoffs from upper TD; full-horizon MC supplies continuation evidence',
            acceptance='after matched full lower training: empirical stability, actual frame coverage, '
                       'unassisted fixed-start physics, local improvement and archived-prefix guards',
            adaptation_clock='separate from main environment steps; actual optimizer schedule state is published',
            adaptation_worlds=self.training.world_count,
            adaptation_voc='incoming lower curriculum stage, held fixed across both branches',
            adaptation_control='one full control curve reused by every window at matched checkpoints')
        for obsolete in ('lower_regularization', 'lower_error_tolerance'):
            manifest.pop(obsolete, None)
        path.write_text(json.dumps(manifest, indent=2)+'\n')

    def _before(self, env, observation, context, action, frame, ids, source):
        # Materialize copies before the next Warp step or donated learner update.
        self.pending = dict(observation=np.array(observation[ids]), context=np.array(context[ids]),
                            action=np.array(action[ids]), frame=np.asarray(frame)[ids].copy(),
                            source=np.full(len(ids), source, np.int32),
                            generation=np.full(len(ids), getattr(self, 'upper_round', 0), np.int32))
        self.pending_ids = ids

    def _after(self, adapter, transition):
        if self.pending is None:
            return
        env, ids = adapter.env, self.pending_ids
        objective = env.objective
        shape = (env.world_count, objective.num_hands, objective.num_bodies, objective.num_basis)
        current = np.array(to_jax(objective.current_wrench_support, shape)[ids])
        reference = np.array(to_jax(objective.wrench_support_target, shape)[ids])
        terminated = np.array(transition.terminated[ids]).astype(bool)
        truncated = np.array(transition.truncated[ids]).astype(bool)
        cost = support_cost(current, reference, tolerance=self.wrench_tolerance)
        row = self.pending
        horizon_end = row['frame'] >= self.binding.num_frames-2
        # A training tracking reset cuts sampling short, not the upper objective.
        # Its terminal observation retains the old reference frame, so it cannot
        # correctly bootstrap the continuation. Keep the row for actor response,
        # omit it from TD, and learn failed suffixes from full-horizon eval MC.
        critic_valid = horizon_end | ~(terminated | truncated)
        row.update(next_observation=np.array(transition.terminal_observation[ids]),
                   next_context=np.array(transition.terminal_critic_context[ids]), cost=cost.astype(np.float32),
                   done=horizon_end.astype(np.float32), critic_valid=critic_valid)
        trial = getattr(self, 'trial_capture', None)
        if trial is None:
            self.replay.add(self.version, **row)
            baseline_trial = getattr(self, 'baseline_capture', None)
            if baseline_trial is not None:
                baseline_trial.add(baseline_trial.reference_key, baseline_trial.policy_key,
                                   world_ids=ids, **row)
        else:
            trial.add(trial.reference_key, trial.policy_key, world_ids=ids, **row)
        self.pending = None

    def before_training_step(self, env, learner, action, step):
        if (step//env.world_count) % self.capture_stride:
            return
        count = min(self.capture_worlds, env.world_count)
        ids = self.hyper_rng.choice(env.world_count, count, replace=False)
        self._before(env, learner.observation, learner.critic_context, action,
                     env.env.timestep.numpy(), ids, 0)

    def after_training_step(self, env, transition):
        self._after(env, transition)

    def before_evaluation_step(self, env, observation, action, frame):
        if self.capture_evaluation or getattr(self, 'trial_capture', None) is not None:
            source = 2 if getattr(self, 'trial_capture', None) is not None else 1
            self._before(env, observation, jnp.array(env.initial_critic_context, copy=True),
                         action, frame, np.arange(env.world_count), source)

    def after_evaluation_step(self, env, transition):
        if self.capture_evaluation or getattr(self, 'trial_capture', None) is not None:
            self._after(env, transition)

    def _evaluate_trial_feedback(self, binding, learner, seed, *, branch, budget, step):
        """Use every tested pair's real unassisted trajectory to correct the critic.

        Trial observations retain their own robot reference and exact inference
        policy. They never enter main-version replay or proposal actor batches.
        This call precedes acceptance so rejected trials provide feedback too.
        """
        if self.capture_evaluation or getattr(self, 'trial_capture', None) is not None:
            raise RuntimeError('nested baseline/trial capture is not supported')
        q = np.ascontiguousarray(binding.joint_q)
        reference_key = hashlib.sha256(str((q.shape, q.dtype.str)).encode()+q.tobytes()).hexdigest()
        trial = TrialRollout(reference_key, actor_policy_digest(learner))
        self.trial_capture = trial
        try:
            result = self.rollout(binding, learner, seed, return_feedback=True)
        finally:
            self.trial_capture = None
            self.pending = None
        self._record_critic_feedback(trial, learner, branch=branch, budget=budget, step=step)
        return result

    def _record_critic_feedback(self, trial, learner, *, branch, budget, step):
        started = time.perf_counter()
        measured = self.hyper.fit_trial_critic(trial, learner, self.hyper_rng)
        row = dict(step=step, branch=branch, budget=budget, environment_steps=budget,
                   reference_sha256=trial.reference_key, policy_sha256=trial.policy_key,
                   critic_fit_seconds=time.perf_counter()-started, **measured)
        self._round_critic_feedback.append(row)
        with (self.output.parent/'critic_feedback.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(f'[upper-feedback] branch={branch} budget={budget} '
              f'rows={measured["trial_feedback_complete_rows"]} '
              f'updates={measured.get("trial_critic_updates", 0)}', flush=True)
        return measured

    def drain_metrics(self):
        metrics, self._metrics = self._metrics, {}
        return metrics

    def _adaptation_branch(self, binding, learner, training_env, stage, seed, budgets, label, step):
        """One extra training scene at a time; preserve its state across checkpoints."""
        import gc
        from flash_chord.envs.rl import RLEnv
        from flash_chord.training.environment import WarpRLEnv
        from flash_chord.retargeting.manipulation_bilevel import assert_device_binding
        config = deepcopy(training_env.env.config)
        config = replace(config, reset=replace(config.reset, seed=seed))
        physical = RLEnv(replace(self.setup.scene, robot_reference=binding), self.setup.reference, config=config)
        assert_device_binding(physical, binding)
        physical.reset(); physical.capture_step(); physical.reset()
        adapter = WarpRLEnv(physical, publish_diagnostics=False, publish_step_metrics=False)
        session = None
        try:
            session = FlashSACAdaptation(adapter, self.training, self.adaptation_config,
                learning_snapshot(learner), stage, seed, executables=self.adaptation_cache)
            self.audit_initial_state(physical, 'adaptation_'+label)
            self.adaptation_cache = session.cache_template()
            previous = 0
            for budget in budgets:
                print(f'[lower-adaptation] {label} target={budget} VOC={stage.voc_scale}', flush=True)
                report = session.advance(budget, physical.timestep.numpy)
                self.adaptation_transitions += budget-previous
                previous = budget
                metrics, _, _, feedback = self._evaluate_trial_feedback(
                    binding, session.state, seed, branch=label, budget=budget, step=step)
                row = dict(environment_steps=budget, cost=self._cost(metrics, binding.joint_q),
                           metrics=metrics, training=report)
                with (self.output.parent/'adaptation_progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=step, branch=label, **row), allow_nan=False)+'\n')
                yield dict(**row, snapshot=learning_snapshot(session.state), feedback=feedback)
        finally:
            if session is not None:
                jax.block_until_ready(session.state.actor.params)
                session.env = session.state = session.action = None
            adapter = physical = None
            gc.collect()

    def _matched_adaptation(self, candidate, learner, env, stage, seed, budgets, controls, label, step):
        comparisons, previous, previous_readiness, stable_count, last = [], None, None, 0, None
        branch = self._adaptation_branch(candidate, learner, env, stage, seed, budgets, label, step)
        try:
            for result in branch:
                control = controls[result['environment_steps']]
                pair = dict(candidate={k: result[k] for k in ('cost', 'metrics', 'training')},
                            control={k: control[k] for k in ('cost', 'metrics', 'training')})
                readiness = training_readiness(result['training'], self.adaptation_config, self.scoring_region)
                # The unchanged controller may keep failing early. That is a
                # valid comparison outcome, not a veto on a candidate that
                # learned to reach the failed interval. Its training budget
                # and actual network updates must still match the protocol.
                readiness += ['control_'+r for r in training_readiness(
                    control['training'], self.adaptation_config, self.scoring_region, require_coverage=False)]
                instability = (['no_previous_ready_checkpoint'] if previous is None else
                    comparison_instability_reasons(previous, pair, self.adaptation_config))
                stable = previous is not None and not readiness and not previous_readiness and not instability
                stable_count = stable_count+1 if stable else 0
                comparisons.append(dict(environment_steps=result['environment_steps'],
                    stable=stable, stable_comparisons=stable_count, readiness_reasons=readiness,
                    instability_reasons=instability, **pair))
                previous, previous_readiness, last = pair, readiness, result
                if stable_count >= self.adaptation_config.stable_comparisons:
                    break
        finally:
            branch.close()
        if last is None:
            raise RuntimeError('adaptation produced no checkpoints')
        stable = stable_count >= self.adaptation_config.stable_comparisons
        reasons = list(comparisons[-1]['readiness_reasons'])
        if not stable:
            reasons.append('adaptation_comparison_not_stable_within_budget')
        return last, controls[last['environment_steps']], dict(checkpoints=comparisons,
            stable=stable, stop_reason='empirical_stability' if stable else 'maximum_budget',
            selected_environment_steps=last['environment_steps'], rejection_reasons=reasons)

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
        minimum = max(2*self.hg_config.batch_size, 256)
        if self.replay.size < minimum:
            self._metrics = {'hypergradient/replay_rows': self.replay.size,
                             'hypergradient/waiting_for_replay': 1}
            return None
        self.upper_round += 1
        started, prior_steps = time.perf_counter(), self.validation_steps
        prior_adaptation = self.adaptation_transitions
        self._round_critic_feedback = []
        seed = self.training.seed+step
        # Supply unassisted physical data even while the lower curriculum uses VOC.
        q = np.ascontiguousarray(self.binding.joint_q)
        reference_key = hashlib.sha256(str((q.shape, q.dtype.str)).encode()+q.tobytes()).hexdigest()
        baseline_trial = TrialRollout(reference_key, actor_policy_digest(learner))
        self.baseline_capture = baseline_trial
        self.capture_evaluation = True
        try:
            baseline, _, _, feedback = self.rollout(self.binding, learner, seed, return_feedback=True)
        finally:
            self.capture_evaluation = False
            self.baseline_capture = None
            self.pending = None
        baseline_progress = self._archive_pair(self.binding, learner,
            np.asarray(feedback['frame_per_object_position_m'])[:, 0],
            np.asarray(feedback['frame_per_object_orientation_rad'])[:, 0], baseline, step, 'baseline')
        context_rows = np.asarray(self.observation(baseline, feedback))
        # Re-score the unchanged incoming pair on the newly chosen local interval.
        baseline, _, _, baseline_feedback = self.rollout(self.binding, learner, seed, return_feedback=True)
        baseline_cost = self._cost(baseline, self.binding.joint_q)
        if self.hyper is None:
            self.hyper = WrenchHypergradient(self.training, env.observation_dim,
                learner.critic_context.shape[-1], context_rows.shape[-1], self.hg_config, seed)
        fit = self.hyper.fit_critic(self.replay, learner, self.hyper_rng)
        # Calibrate on this exact unassisted incoming controller before asking
        # for any proposal; old reference-version rollouts are not its rollout.
        baseline_fit = self._record_critic_feedback(
            baseline_trial, learner, branch='baseline', budget=0, step=step)
        for window in self.window_contexts:
            window['candidate_count'] = 1
        windows, trials, best = list(self.window_contexts), [], None
        before_q = self.binding.joint_q.copy()
        budget_report = self.adaptation_config.budget_report(
            env.world_count, self.binding.num_frames, region=self.scoring_region)
        budgets = tuple(budget_report['environment_steps'])
        if budgets[0] <= max(self.adaptation_config.replay_minimum_size, self.training.replay.batch_size):
            raise ValueError('first adaptation checkpoint must allow learning after replay warmup')
        print(f'[lower-adaptation] resolved matched budgets={budgets} worlds={env.world_count}', flush=True)
        controls = None
        for window in windows:
            representative = int(np.flatnonzero(self.window_ids == window['window_id'])[0])
            self.select_candidate_context(representative)
            active = np.flatnonzero(self.basis[:, 0] > 0)
            if not len(active):
                trials.append(dict(window=window, accepted=False, skipped='no editable interior frames'))
                continue
            frame_range = (int(active[0]), int(active[-1])+1)
            try:
                validation, _ = self.replay.sample(self.hyper_rng, self.hg_config.batch_size,
                    frame_range=frame_range, source=1, generation=self.upper_round)
                batch, _ = self.replay.sample(self.hyper_rng, self.hg_config.batch_size*self.hg_config.inner_steps,
                                              frame_range=frame_range, source=0)
            except ValueError as error:
                trials.append(dict(window=window, accepted=False, skipped=str(error)))
                continue
            print(f'[wrench-hypergradient] step={step} window={window["window_id"]} reference linearization', flush=True)
            relabel = ReferenceRelabel.from_editor(self, env.env.observation_strategy)
            relabel_error = max(relabel.validate_zero(batch['observation'], batch['frame']),
                                relabel.validate_zero(validation['observation'], validation['frame']))
            batch['noise'] = jnp.asarray(self.hyper_rng.normal(size=(len(batch['frame']), 56)), jnp.float32)
            batch = jax.tree.map(lambda x: x.reshape((self.hg_config.inner_steps, self.hg_config.batch_size, *x.shape[1:])), batch)
            theta, _, _, diagnostics = self.hyper.propose(
                context_rows[representative], learner, batch, validation, relabel)
            candidate = binding_for_q(self, self.candidate_joint_q(theta))
            # Both physical branches start from the SAME complete incoming
            # learner. The three-step copies only guided the proposal gradient.
            if controls is None:
                controls = {}
                for result in self._adaptation_branch(self.binding, learner, env, stage,
                                                      seed, budgets, 'noedit', step):
                    controls[result['environment_steps']] = {k: result[k] for k in ('cost', 'metrics', 'training')}
            result, control, adaptation = self._matched_adaptation(candidate, learner, env, stage,
                seed, budgets, controls, f'window_{window["window_id"]}', step)
            after, after_feedback = result['metrics'], result['feedback']
            cost, control_cost = result['cost'], control['cost']
            control_rejections = self._local_rejections(after, control['metrics'])
            baseline_rejections = self._local_rejections(after, baseline)
            reasons = adaptation['rejection_reasons'] + control_rejections + baseline_rejections
            if cost >= min(control_cost, baseline_cost)-1e-4:
                reasons.append('no_improvement_over_adapted_noedit_and_incoming_pair')
            row = dict(window=window, accepted=not reasons, rejection_reasons=sorted(set(reasons)),
                       cost=cost, control_cost=control_cost, baseline_cost=baseline_cost,
                       actual_local_cost_gain=control_cost-cost, relabel_zero_error=relabel_error,
                       validation_generation=self.upper_round,
                       control_guard_reasons=control_rejections, baseline_guard_reasons=baseline_rejections,
                       theta=theta.tolist(), metrics=after, adaptation=adaptation, **diagnostics)
            trials.append(row)
            if not reasons and (best is None or cost < best[0]):
                best = (cost, candidate, result['snapshot'], after, after_feedback, row)
        accepted, replacement, updated = best is not None, None, learner
        selected, published, published_feedback = None, baseline, baseline_feedback
        old_version = self.version
        if accepted:
            _, candidate, snapshot, published, published_feedback, selected = best
            updated = install_learning(learner, snapshot)
            from flash_chord.envs.rl import RLEnv
            from flash_chord.training.environment import WarpRLEnv
            from flash_chord.retargeting.manipulation_bilevel import assert_device_binding
            replacement = RLEnv(replace(self.setup.scene, robot_reference=candidate),
                                 self.setup.reference, config=deepcopy(env.env.config))
            assert_device_binding(replacement, candidate)
            replacement.reset(); replacement.capture_step(); replacement.reset()
            replacement = WarpRLEnv(replacement, publish_diagnostics=False, publish_step_metrics=False)
            self.binding, self.version = candidate, self.version+1
            self.publish(step)
            self.replay.clear(self.version)
        if command_digest(self.setup.reference) != self.command_sha256:
            raise RuntimeError('hypergradient editor mutated fixed object/contact targets')
        progress = self._archive_pair(self.binding, updated,
            np.asarray(published_feedback['frame_per_object_position_m'])[:, 0],
            np.asarray(published_feedback['frame_per_object_orientation_rad'])[:, 0],
            published, step, 'accepted' if accepted else 'retained', save_pair=accepted)
        change = {}
        for hand in self.projector.layout.hands:
            ids = [qid for name, qid in zip(self.projector.layout.scalar_joints.names, self.projector.qids)
                   if name.startswith(hand.side+'_') and name.endswith(('wrist_x', 'wrist_y', 'wrist_z'))]
            for name, origin in [('accepted_increment', before_q), ('cumulative', self.baseline.joint_q)]:
                delta = self.binding.joint_q-origin
                change[f'{name}_{hand.side}_wrist_peak_cm'] = float(np.linalg.norm(delta[:, ids], axis=-1).max()*100)
                change[f'{name}_{hand.side}_finger_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean(delta[:, hand.finger_q_ids]**2))))
        record = dict(engine='wrench_hypergradient', step=step, upper_round=self.upper_round,
            accepted=accepted, version_before=old_version, version_after=self.version,
            failure_context=self.failure_context, windows=windows, selected_window=selected['window'] if selected else None,
            candidates=trials, baseline_cost=baseline_cost, progress=progress, baseline_progress=baseline_progress,
            baseline_metrics=baseline, baseline_critic=baseline_fit,
            outer_validation_generation=self.upper_round,
            reference_change=change, critic=fit, seconds=time.perf_counter()-started,
            upper_simulation_transitions=self.validation_steps-prior_steps,
            adaptation_transitions=self.adaptation_transitions-prior_adaptation,
            adaptation_checkpoints=list(budgets), adaptation_budget=budget_report,
            critic_feedback=list(self._round_critic_feedback),
            adaptation_voc_scale=float(stage.voc_scale),
            lower_response='matched_full_FlashSAC_fresh_experience',
            gradient_estimator=f'short_actor_{self.hg_config.inner_optimizer}_unrolling_proposal_then_bounded_adaptation_validation')
        path = self.output.parent/'search_batches'/f'step_{step}'
        path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, allow_nan=False)+'\n'
        (path/'scores.json').write_text(payload)
        save_snapshot(path/'selected_reference.npz', self.binding, self.projector.layout,
                      provenance=dict(step=step, accepted=accepted, engine='wrench_hypergradient'))
        (path/'upper_state.msgpack').write_bytes(serialization.to_bytes(dict(
            actor=self.hyper.upper_params, actor_optimizer=self.hyper.upper_state,
            critic=self.hyper.cost_params, target_critic=self.hyper.target_params,
            critic_optimizer=self.hyper.cost_state)))
        for name in ('upper_updates.jsonl', 'adaptive_search_summary.jsonl'):
            with (self.output.parent/name).open('a') as stream:
                stream.write(payload)
        self._metrics = {f'hypergradient/{k}': float(v) for k, v in fit.items()}
        self._metrics.update({'hypergradient/accepted': float(accepted),
                              'hypergradient/physics_trials': len(trials),
                              'adaptation/environment_transitions': float(record['adaptation_transitions']),
                              'adaptation/max_budget_per_branch': float(budgets[-1]),
                              'adaptation/selected_budget': float(selected['adaptation']['selected_environment_steps']) if selected else 0.,
                              'adaptation/stable_candidates': sum(t.get('adaptation', {}).get('stable', False) for t in trials)})
        feedback_rows = self._round_critic_feedback
        feedback_fits = [row for row in feedback_rows if not row.get('trial_critic_skipped', 1)]
        self._metrics.update({
            'upper_feedback/complete_rows': sum(row['trial_feedback_complete_rows'] for row in feedback_rows),
            'upper_feedback/critic_updates': sum(row.get('trial_critic_updates', 0) for row in feedback_rows),
            'upper_feedback/tested_checkpoints': len(feedback_rows),
            'upper_feedback/fit_seconds': sum(row['critic_fit_seconds'] for row in feedback_rows),
        })
        if feedback_fits:
            for key in ('heldout_return_mae_before', 'heldout_return_mae_after',
                        'parameter_delta_l2', 'prediction_before', 'prediction_after', 'measured_return'):
                self._metrics[f'upper_feedback/{key}'] = float(np.mean([
                    row[f'trial_critic_{key}'] for row in feedback_fits]))
        measured = [t for t in trials if 'gradient_norm' in t]
        if measured:
            for key in ('gradient_norm', 'direct_gradient_norm', 'response_gradient_norm',
                        'predicted_wrench_return_gain', 'actual_local_cost_gain', 'response_gradient_fraction'):
                self._metrics[f'hypergradient/{key}'] = float(np.mean([t[key] for t in measured]))
        self._metrics.update({f'reference_change/{k}': v for k, v in change.items()})
        print(f'[wrench-hypergradient] accepted={accepted} trials={len(trials)} seconds={record["seconds"]:.1f}', flush=True)
        return ReferenceLearnerUpdate(updated, replacement)
