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
        if options.get('defer_acceptance_gates_to_confirmation',False) and (
                not hasattr(self,'confirm_candidate') or not options.get('robust_confirmation',True)):
            raise ValueError('deferred screening requires paired robust confirmation')
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

    def __call__(self, env, learner, step, stage):
        if step < self.next_update or step >= self.training.total_environment_steps:
            return None
        self.next_update += self.interval
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
            if hasattr(self, 'acceptance_gate'):
                return self.acceptance_gate(metrics, control)
            return (metrics['all_errors_finite'] and metrics['full_horizon']
                and passes_study_gate(metrics, control)
                and (not self.contact_proximity_weight or contact_proximity_gate(metrics, control)))
        baseline_cost = cost(baseline, self.binding.joint_q)
        obs = self.observation(baseline, feedback)
        if obs.shape[0] == 1:
            obs = jnp.repeat(obs, self.upper_worlds, axis=0)
        if obs.shape[0] != self.upper_worlds:
            raise ValueError('one upper observation per candidate world required')
        action, context = self.prepare_upper(obs)
        if self.guided_contact_proposal:
            guided = contact_directed_action(feedback['contact_direction_bins'],
                list(self.projector.layout.scalar_joints.names), self.projector.qids,
                self.joint_basis, float(self.options.get('wrist_translation_offset_m', .03)))
            action = action.at[0].set(jnp.asarray(guided))
        if self.options.get('include_noop', False):
            action = action.at[0].set(0.)
        actions = np.asarray(action)
        qs, targets, rejections = [], [], {}
        for world, proposal in enumerate(actions):
            if hasattr(self, 'select_candidate_context'):
                self.select_candidate_context(world)
            try:
                q = self.binding.joint_q.copy() if world == 0 and self.options.get('include_noop',False) else self.candidate_joint_q(proposal)
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
        selection_mask = valid.copy()
        if self.options.get('defer_acceptance_gates_to_confirmation', False):
            # Single-world and batched contact solvers differ slightly; enforce
            # strict acceptance only on the paired confirmation cohort.
            selection_mask = np.asarray([i not in rejections and m['all_errors_finite'] and m['full_horizon'] and np.isfinite(c)
                for i,(m,c) in enumerate(zip(metrics,costs))])
        if self.options.get('include_noop',False): selection_mask[0] = False
        ranked = np.argsort(np.where(selection_mask, costs, np.inf))
        shortlist = [int(i) for i in ranked if selection_mask[i] and costs[i]<baseline_cost][:int(self.options.get('confirmation_top_k',1))]
        if hasattr(self, 'shortlist_candidates'):
            shortlist = self.shortlist_candidates(ranked, selection_mask, costs, baseline_cost)
        selected = shortlist[0] if shortlist else None
        accepted = False
        record = dict(step=step, engine='parallel_flashsac', upper_round=self.upper_round,
            version_before=self.version, baseline=baseline, lower_policy_frozen=True,
            candidate_worlds=self.upper_worlds, replicas_per_candidate=1,
            object_contact_sha256=self.command_sha256,
            candidates=[dict(trial=i,cost=float(c),reward=float(r),metrics=m,gate_passed=bool(v),
                             rejected=rejections.get(i)) for i,(c,r,m,v) in enumerate(zip(costs,rewards,metrics,valid))])
        record['gate_diagnostics'] = dict(
            improving_candidates=int(np.sum(costs < baseline_cost)),
            task_gate_pass=int(sum(passes_study_gate(m,baseline) for m in metrics)),
            contact_gate_pass=int(sum(contact_proximity_gate(m,baseline) for m in metrics)),
            all_gates_pass=int(valid.sum()), baseline_cost=float(baseline_cost),
            noop_cost=float(costs[0]) if self.options.get('include_noop',False) else None)
        proposal_diagnostics = self.proposal_diagnostics(actions,qs) if hasattr(self, 'proposal_diagnostics') else {}
        evidence = self.output.parent/'search_batches'/f'step_{step}'
        evidence.mkdir(parents=True,exist_ok=True)
        # Preserve all proposals for reproducible later comparisons, not only the winner.
        np.savez_compressed(evidence/'proposals.npz',actions=actions,joint_q=qs,joint_target=targets,costs=costs)
        record['confirmation_attempts'] = []
        for selected in shortlist:
            if hasattr(self, 'select_candidate_context'):
                self.select_candidate_context(selected)
            candidate = build_robot_reference_binding(self.projector.builder,self.projector.layout,
                qs[selected],targets[selected],self.baseline.fps)
            check, _, _ = self.rollout(candidate,learner,seed)
            accepted = gate(check,baseline) and cost(check,candidate.joint_q) < baseline_cost
            if hasattr(self, 'confirm_candidate') and self.options.get('robust_confirmation', True):
                accepted, record['robust_confirmation'] = self.confirm_candidate(candidate,learner,seed,baseline,check)
            save_snapshot(evidence/'selected_reference.npz',candidate,self.setup.scene.layout,
                          provenance=dict(step=step,trial=selected,accepted=bool(accepted),object_contact_sha256=self.command_sha256))
            record['validation'] = dict(baseline=baseline,candidate=check,
                proposal_source=f'upper_trial_{selected}',deterministic_confirmation=True)
            record['confirmation_attempts'].append(dict(trial=selected,nominal=record['validation'],
                robust=record.get('robust_confirmation'),accepted=bool(accepted)))
            if accepted:
                self.binding=candidate;self.version+=1;self.publish(step)
                break
        save_learner_checkpoint(self.output.parent/'upper',self.upper,
            metadata={'lower_step':step,'engine':'parallel_flashsac'},save_replay=True)
        if command_digest(self.setup.reference)!=self.command_sha256:
            raise RuntimeError('upper editor mutated immutable object/contact commands')
        record.update(proposal_diagnostics)
        record.update(accepted=bool(accepted),version_after=self.version,selected_trial=selected,
            upper_optimizer_updates=int(self.upper.global_update_step),seconds=time.perf_counter()-started,
            upper_simulation_transitions=self.validation_steps-previous_steps,
            cumulative_upper_simulation_transitions=self.validation_steps)
        if hasattr(self, 'finalize_search_record'):
            self.finalize_search_record(record, qs, costs, valid)
        # Reject nonfinite diagnostic records instead of publishing corrupt JSON.
        payload=json.dumps(record,allow_nan=False)+'\n'
        (evidence/'scores.json').write_text(payload)
        with (self.output.parent/'upper_updates.jsonl').open('a') as stream:stream.write(payload)
        print(f'[parallel] step={step} accepted={accepted} seconds={record["seconds"]:.1f}',flush=True)
        # The runner interprets a non-None return as a changed reference and
        # resets all worlds and lower replay. A rejected search changes neither.
        if not accepted:return None
        replacement=RLEnv(replace(self.setup.scene,robot_reference=self.binding),self.setup.reference,config=deepcopy(env.env.config))
        assert_device_binding(replacement,self.binding)
        np.testing.assert_array_equal(replacement.command.reference.body_pos_w.numpy(),env.env.command.reference.body_pos_w.numpy())
        np.testing.assert_array_equal(replacement.command.reference.body_quat_w.numpy(),env.env.command.reference.body_quat_w.numpy())
        replacement.reset();replacement.capture_step();replacement.reset()
        return WarpRLEnv(replacement,publish_diagnostics=False,publish_step_metrics=False)
