"""Nested reference optimization with solved convex policy-head responses.

Every search stage defines an explicit bilevel surrogate around its current
FlashSAC actor/critic. Its lower error is checked, and only evaluated reference
AND policy pairs can be published. Global neural-RL convergence is not claimed.
"""
from copy import deepcopy
from dataclasses import replace
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
from flash_chord.retargeting.parallel_rollout import rollout_candidates
from flash_chord.retargeting.response_policy import ConvexHeadAdapter
from flash_chord.retargeting.reference_snapshot import save_snapshot,command_digest
from flash_chord.retargeting.ttsa_editor import passes_study_gate
from flash_chord.retargeting.chord_upper import WrenchSupportAccumulator,wrench_contact_gate
from flash_chord.retargeting.failure_windows import prefix_gate
from flash_chord.retargeting.manipulation_bilevel import assert_device_binding
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.reference_update import ReferenceLearnerUpdate
from flash_chord.training.flash_sac.learner import _create_adam
from flash_chord.retargeting.local_acceptance import (
    scoring_region, local_pose_metrics, LocalWrenchAccumulator, local_guard_reasons,
)


def joint_object_metrics(position,orientation):
    p,a=np.asarray(position),np.asarray(orientation)
    return dict(object_success_robust_fraction=float(np.mean(np.all((p<.1)&(a<.5),axis=-1))),
                unstable_world_fraction=float(np.any(p>1.)))


def geometric_edit_penalty(joint_q, baseline, layout, fps):
    """Separate dimensionless translation/finger/SO(3) edit and rate penalties."""
    terms=[]; rates=[]
    for hand in layout.hands:
        # Floating Sharpa layout uses wrist xyz followed by an xyzw quaternion.
        # Finger q ids identify the end of its seven-coordinate wrist block.
        start=int(min(hand.finger_q_ids))-7
        translation=joint_q[:,start:start+3]-baseline[:,start:start+3]
        rotation=(Rotation.from_quat(joint_q[:,start+3:start+7])*
                  Rotation.from_quat(baseline[:,start+3:start+7]).inv()).as_rotvec()
        finger=joint_q[:,hand.finger_q_ids]-baseline[:,hand.finger_q_ids]
        for value,scale,speed in [(translation,.03,.1),(rotation,.52,1.5),(finger,.52,2.)]:
            terms.append(float(np.mean((value/scale)**2)))
            rates.append(float(np.mean((np.diff(value,axis=0)*fps/speed)**2)))
    return .05*float(np.mean(terms))+.05*float(np.mean(rates))


def response_cost_terms(metrics, joint_q, baseline, layout, fps, wrench_weight=5., *, local=False):
    prefix = 'local_' if local else ''
    return dict(object_position=metrics[prefix+'object_position_robust_m']/.03,
        object_orientation=metrics[prefix+'object_orientation_robust_rad']/.3,
        joint_object_failure=2*(1-metrics[prefix+'object_success_robust_fraction']),
        wrench_support=wrench_weight*metrics[prefix+'cws_tolerance_error'],
        missed_contact=metrics[prefix+'cws_missed_fraction'],
        unintended_contact=metrics[prefix+'cws_unintended_fraction']+metrics[prefix+'cws_unintended_energy'],
        instability=3*metrics['unstable_world_fraction'],
        reference_regularization=geometric_edit_penalty(joint_q,baseline,layout,fps))


def response_cost(metrics, joint_q, baseline, layout, fps, wrench_weight=5., *, local=False):
    raw=sum(response_cost_terms(metrics,joint_q,baseline,layout,fps,wrench_weight,local=local).values())
    if not np.isfinite(raw):raise FloatingPointError('nonfinite upper cost')
    # Defines a bounded empirical upper objective, rather than silently clipping
    # only selected candidates or using different scales for confirmation.
    return float(np.clip(raw,0.,100.))


class ResponseBilevelEditor(AdaptiveWindowEditor):
    response_metrics=staticmethod(joint_object_metrics)

    def __init__(self, **kwargs):
        if float(kwargs['options'].get('contact_proximity_weight',0.)) != 0:
            raise ValueError('CHORD response search must not use contact-location penalties')
        super().__init__(**kwargs)
        self.acceptance_mode = self.options.get('acceptance_mode', 'strict_robust')
        if self.acceptance_mode not in ('strict_robust', 'local_nominal'):
            raise ValueError('unknown reference acceptance mode')
        self.local_acceptance = self.acceptance_mode == 'local_nominal'
        self.scoring_region = None
        self.fixed_start = bool(self.options.get('fixed_start',False))
        self.initial_state_sha256 = None
        self.demo_archive = None
        if self.fixed_start:
            if not self.local_acceptance or int(self.options.get('failure_baseline_worlds',0)):
                raise ValueError('fixed-start reconstruction needs local acceptance and an unperturbed baseline')
            reset=self.config['env']['reset']
            if not reset.get('fixed_initial_state') or reset.get('voc_decay_steps')!=0:
                raise ValueError('fixed-start training must reset exact frame-zero joints without settling')
            from flash_chord.retargeting.demo_archive import DemoArchive
            self.demo_archive=DemoArchive(self)
        self.proposal_mode = self.options.get('upper_proposal', 'gaussian')
        if self.proposal_mode not in ('gaussian', 'flashsac'):
            raise ValueError('upper_proposal must be gaussian or flashsac')
        self.proposal_learner = None
        if self.proposal_mode == 'flashsac':
            from flash_chord.retargeting.response_upper import ResponseUpperFlashSAC
            self.proposal_learner = ResponseUpperFlashSAC(self.training.seed+20000, self.upper_worlds,
                updates=int(self.options.get('upper_updates', 32)),
                initial_std=float(self.options.get('upper_initial_std', .5)))
        self.wrench_weight=float(self.options.get('cws_weight',5.))
        self.wrench_tolerance=float(self.options.get('cws_tolerance',.1))
        if not np.isfinite(self.wrench_weight) or self.wrench_weight <= 0:
            raise ValueError('positive finite wrench weight required')
        if self.contact_proximity is not None:
            raise ValueError('contact-location guidance must be disabled')
        self.response_adapter=ConvexHeadAdapter(self.training,
            regularization=float(self.options.get('response_ridge',1.)),
            teacher_step=float(self.options.get('response_teacher_step',.1)),
            target_clip=float(self.options.get('response_target_clip',.1)),
            error_tolerance=float(self.options.get('response_error_tolerance',1e-5)),
            sample_stride=int(self.options.get('response_sample_stride',4)))
        self.response_rng=np.random.default_rng(self.training.seed+47000)
        self.baseline_probe = None
        probe_worlds = int(self.options.get('failure_baseline_worlds', 0))
        if probe_worlds:
            from flash_chord.retargeting.baseline_probe import BaselineFailureProbe
            self.baseline_probe = BaselineFailureProbe(self, embodiment=kwargs['embodiment'],
                collision=kwargs['collision'], worlds=probe_worlds)
        path=self.output.parent/'ttsa_manifest.json'
        manifest=json.loads(path.read_text())
        manifest.update(engine='response_bilevel',lower_response='candidate_conditioned_convex_mean_head',
            lower_objective='ridge fit of bounded frozen-critic SAC logit-improvement targets',
            lower_regularization=self.response_adapter.regularization,
            lower_error_tolerance=self.response_adapter.error_tolerance,
            guarantee_scope='lower surrogate residual bound and finite evaluated upper set; not full neural RL convergence',
            upper_learning=self.proposal_mode,
            upper_updates_per_batch=int(self.options.get('upper_updates',32)) if self.proposal_learner else 0,
            pair_publication=True,confirmation='all per-reset object, prefix, missed/unintended wrench gates required',
            upper_contact_objective='CHORD-style normalized squared tolerance-band support loss',
            cws_weight=self.wrench_weight,cws_tolerance=self.wrench_tolerance,
            contact_location_penalty=False,contact_location_gate=False,
            cws_contact_regression_tolerance=float(self.options.get('cws_contact_regression_tolerance',.02)))
        manifest['acceptance_mode'] = self.acceptance_mode
        if self.fixed_start:
            manifest.update(fixed_initial_state=True,window_selection='fixed_start_nominal_failure',
                failure_baseline_worlds=1,failure_baseline_perturbations=False,
                historical_prefix_protection=True,best_demo_pointer=str(self.output.parent/'best_demo.json'))
        if self.local_acceptance:
            manifest.update(confirmation='nominal demonstration start; no perturbed-reset acceptance veto',
                cws_contact_regression_tolerance=None,
                upper_objective_scope='shared error-onset to failure plus lookahead interval, both objects',
                acceptance_lookahead_seconds=float(self.options.get('acceptance_lookahead_seconds',1.)),
                large_position_regression_m=float(self.options.get('acceptance_position_regression_m',.05)),
                large_orientation_regression_deg=float(self.options.get('acceptance_orientation_regression_deg',20.)))
        if self.baseline_probe is not None:
            manifest.update(window_selection='baseline_onset_consensus_and_sampled_history',
                failure_baseline_worlds=probe_worlds, failure_baseline_parallel=True,
                failure_consensus_fraction=float(self.options.get('failure_consensus_fraction', .5)),
                failure_consensus_cluster_seconds=float(self.options.get('failure_consensus_cluster_seconds', .5)))
        if self.proposal_learner:
            manifest['upper_flashsac_config'] = self.proposal_learner.configuration()
        path.write_text(json.dumps(manifest,indent=2)+'\n')

    def audit_initial_state(self,env,context):
        if not self.fixed_start:return
        from flash_chord.retargeting.fixed_start import assert_initial_state
        result=assert_initial_state(env,self.initial_state_sha256)
        self.initial_state_sha256=result['initial_state_sha256']
        result.update(context=context,reference_version=self.version,
                      voc_scale=float(env.object_control_scale.numpy()[0]))
        with (self.output.parent/'initial_state_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')

    def _archive_pair(self,binding,learner,p,a,metrics,step,origin,*,save_pair=False):
        if self.demo_archive is None:return None
        cost=response_cost(metrics,binding.joint_q,self.baseline.joint_q,self.projector.layout,
                           self.baseline.fps,self.wrench_weight)
        return self.demo_archive.consider(binding,learner,p,a,metrics,cost,step,origin,save_pair=save_pair)

    def on_checkpoint(self,env,learner,step):
        if not self.fixed_start:return
        metrics,_,_,feedback=self.rollout(self.binding,learner,self.training.seed+step,return_feedback=True)
        p=np.asarray(feedback['frame_per_object_position_m'])[:,0]
        a=np.asarray(feedback['frame_per_object_orientation_rad'])[:,0]
        report=self._archive_pair(self.binding,learner,p,a,metrics,step,'checkpoint')
        print('[fixed-start-checkpoint] '+json.dumps(report),flush=True)

    def new_wrench_accumulator(self,env):
        objective=env.objective
        if self.local_acceptance and self.scoring_region is not None:
            return LocalWrenchAccumulator(env.world_count,objective.num_hands,objective.num_bodies,
                objective.num_basis,tolerance=self.wrench_tolerance,region=self.scoring_region)
        return WrenchSupportAccumulator(env.world_count,objective.num_hands,objective.num_bodies,
                                       objective.num_basis,tolerance=self.wrench_tolerance)

    def observation(self, metrics, feedback=None):
        observation = super().observation(metrics, feedback)
        if self.fixed_start:
            self.failure_context.update(detector='fixed_start_nominal',baseline_worlds=1,
                baseline_probe_distinct_resets=1,baseline_probe_translation_m=0.,baseline_probe_finger_degrees=0.,
                baseline_probe_voc_scale=0.,initial_state_sha256=self.initial_state_sha256)
        if self.local_acceptance:
            self.scoring_region = scoring_region(self.failure_context, self.baseline.num_frames,
                self.baseline.fps, lookahead_seconds=float(self.options.get('acceptance_lookahead_seconds',1.)))
            self.protected_prefix_frames = self.scoring_region['start_frame']
            print('[local-acceptance] '+json.dumps(self.scoring_region), flush=True)
        return observation

    def trajectory_metrics(self, p, a):
        metrics = super().trajectory_metrics(p, a)
        if getattr(self, 'local_acceptance', False) and self.scoring_region is not None:
            metrics.update(local_pose_metrics(p, a, self.scoring_region))
        if getattr(self,'demo_archive',None) is not None:
            metrics.update(self.demo_archive.check(p,a))
        return metrics

    def rollout(self,*args,**kwargs):
        # The inherited single-world rollout exposes per-frame object feedback.
        requested=kwargs.get('return_feedback',False)
        kwargs['return_feedback']=True
        metrics,achieved,commanded,feedback=super().rollout(*args,**kwargs)
        p=np.asarray(feedback['frame_per_object_position_m'])
        a=np.asarray(feedback['frame_per_object_orientation_rad'])
        # [frame,world,object], with one world in this evaluator.
        metrics.update(joint_object_metrics(p,a))
        return (metrics,achieved,commanded,feedback) if requested else (metrics,achieved,commanded)

    def _cost(self,m,q):
        return response_cost(m,q,self.baseline.joint_q,self.projector.layout,
                             self.baseline.fps,self.wrench_weight,local=self.local_acceptance)

    def _cost_terms(self,m,q):
        return response_cost_terms(m,q,self.baseline.joint_q,self.projector.layout,
                                   self.baseline.fps,self.wrench_weight,local=self.local_acceptance)

    def _local_rejections(self,after,before):
        reasons=local_guard_reasons(after,before,
            position_regression_m=float(self.options.get('acceptance_position_regression_m',.05)),
            orientation_regression_rad=np.deg2rad(float(self.options.get('acceptance_orientation_regression_deg',20.))))
        if self.fixed_start:reasons.extend(after['saved_prefix_rejection_reasons'])
        return reasons

    def _gate(self,after,before):
        if getattr(self, 'local_acceptance', False):
            return not self._local_rejections(after,before)
        return (after['all_errors_finite'] and before['all_errors_finite']
                and after['full_horizon'] and before['full_horizon']
                and passes_study_gate(after,before) and prefix_gate(after,before)
                and wrench_contact_gate(after,before,
                    tolerance=float(self.options.get('cws_contact_regression_tolerance',.02))))

    def _learner_for(self,learner,responses,index):
        params=responses.params_for(learner.actor.params,index)
        actor=learner.actor.replace(params=params,optimizer_state=_create_adam(self.training).init(params))
        return learner.replace(actor=actor)

    def __call__(self,env,learner,step,stage):
        if step<self.next_update or step>=self.training.total_environment_steps:return None
        self.next_update+=self.interval;self.upper_round+=1
        jax.block_until_ready(learner)
        started=time.perf_counter();previous_steps=self.validation_steps
        seed=self.training.seed+step
        baseline,_,_,feedback=self.rollout(self.binding,learner,seed,return_feedback=True)
        baseline_progress=self._archive_pair(self.binding,learner,
            np.asarray(feedback['frame_per_object_position_m'])[:,0],
            np.asarray(feedback['frame_per_object_orientation_rad'])[:,0],baseline,step,'baseline')
        if self.baseline_probe is not None:
            probe_seed = int(np.random.SeedSequence([self.training.seed, self.upper_round, 81001]).generate_state(1)[0])
            p, a, deltas, probe_metadata = self.baseline_probe(self, learner, probe_seed)
            feedback.update(baseline_ensemble_position_m=p, baseline_ensemble_orientation_rad=a,
                            baseline_ensemble_metadata=probe_metadata)
            probe_path = self.output.parent/'search_batches'/f'step_{step}'
            probe_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probe_path/'baseline_ensemble.npz', position_m=p, orientation_rad=a,
                                initial_joint_delta=deltas, seed=probe_seed)
        observation = self.observation(baseline,feedback)
        # Both modes have Gaussian support; the SAC mode learns its contextual
        # mean/covariance. Feasible projection and lower responses are identical.
        if self.proposal_learner:
            actions = self.proposal_learner.propose(observation)
        else:
            actions=self.response_rng.normal(0.,float(self.options.get('upper_initial_std',.5)),
                                              (self.upper_worlds,56)).astype(np.float32)
            actions[0]=0.
        qs=[];targets=[]
        for i,action in enumerate(actions):
            self.select_candidate_context(i)
            q=self.binding.joint_q.copy() if i==0 else self.candidate_joint_q(action)
            target=self.baseline.joint_target.copy()
            target[:,self.projector.dids]=q[:,self.projector.qids]
            for qi,di in self.projector.ball_joints:
                target[:,di:di+3]=Rotation.from_quat(q[:,qi:qi+4]).as_rotvec()
            qs.append(q);targets.append(target)
        qs,targets=np.asarray(qs),np.asarray(targets)
        collect,rows=self.response_adapter.collector(learner,seed)
        print(f'[response-bilevel] step={step} collecting lower data for {len(qs)} candidates',flush=True)
        frozen_metrics,_,_=rollout_candidates(self,learner,qs,targets,on_observation=collect)
        responses=self.response_adapter.fit(learner,rows)
        del rows
        print('[response-bilevel] lower parameter error bound max='+
              str(float(responses.certificate.parameter_error_bound.max())),flush=True)
        candidate_feedback={}
        adapted_metrics,_,_=rollout_candidates(self,learner,qs,targets,
            policy_override=self.response_adapter.policy_for(learner,responses),
            on_feedback=lambda p,a:candidate_feedback.update(position=p,orientation=a))
        if not (adapted_metrics[0]['all_errors_finite'] and adapted_metrics[0]['full_horizon']):
            raise FloatingPointError('adapted no-edit control did not produce a valid full-horizon rollout')
        costs=np.asarray([self._cost(m,q) for m,q in zip(adapted_metrics,qs)])
        frozen_costs=np.asarray([self._cost(m,q) for m,q in zip(frozen_metrics,qs)])
        upper_learning = self.proposal_learner.learn(costs) if self.proposal_learner else dict(upper_optimizer_updates=0)
        control_learner=self._learner_for(learner,responses,0)
        eligible=np.asarray([self._gate(m,adapted_metrics[0]) for m in adapted_metrics])
        eligible[0]=False
        ranked=np.argsort(np.where(eligible,costs,np.inf))
        finalists=self.shortlist_candidates(ranked,eligible,costs,costs[0])
        attempts=[];feasible=[]
        if self.local_acceptance:
            # The vectorized batch already evaluated each reference/head pair
            # from the same original demonstration start, including no-edit.
            # Rank on one common local interval; keep full-horizon guards only.
            control_mean=float(costs[0])
            viable=[int(i) for i in ranked if eligible[i] and costs[i]<costs[0]-1e-4]
            if viable:
                index=viable[0]
                candidate=build_robot_reference_binding(self.projector.builder,self.projector.layout,
                                                         qs[index],targets[index],self.baseline.fps)
                feasible.append((float(costs[index]),index,candidate,self._learner_for(learner,responses,index)))
        else:
            # Validation uses one unchanged-reference SOLVED response as the
            # control, not the old frozen actor or a mismatched baseline world.
            reset_seeds=np.random.SeedSequence([self.training.seed,self.upper_round,71001]).generate_state(3)
            controls=[];deltas=[]
            for reset_seed in reset_seeds:
                rng=np.random.default_rng(int(reset_seed));delta=np.zeros(self.baseline.num_joint_q)
                for name,qid in zip(self.projector.layout.scalar_joints.names,self.projector.qids):
                    bound=.001 if name.endswith(('wrist_x','wrist_y','wrist_z')) else np.deg2rad(.5)
                    delta[qid]=rng.uniform(-bound,bound)
                deltas.append(delta)
                before,_,_=self.rollout(self.binding,control_learner,seed,initial_q_delta=delta)
                controls.append(before)
            control_mean=float(np.mean([self._cost(m,qs[0]) for m in controls]))
            for index in finalists:
                candidate=build_robot_reference_binding(self.projector.builder,self.projector.layout,
                                                         qs[index],targets[index],self.baseline.fps)
                candidate_learner=self._learner_for(learner,responses,index)
                checks=[]
                for reset_seed,delta,before in zip(reset_seeds,deltas,controls):
                    after,_,_=self.rollout(candidate,candidate_learner,seed,initial_q_delta=delta)
                    checks.append(dict(seed=int(reset_seed),baseline=before,candidate=after,
                        gate_passed=bool(self._gate(after,before)),cost=self._cost(after,qs[index]),
                        cost_gain=self._cost(before,qs[0])-self._cost(after,qs[index])))
                mean_cost=float(np.mean([c['cost'] for c in checks]))
                passed=(all(c['gate_passed'] for c in checks) and mean_cost<control_mean
                        and sum(c['cost_gain']>0 for c in checks)>=2)
                attempts.append(dict(trial=int(index),accepted=bool(passed),mean_cost=mean_cost,checks=checks))
                if passed:feasible.append((mean_cost,int(index),candidate,candidate_learner))
        # Prefer the solved no-edit response when no reference edit is feasible.
        # Fixed-start runs may reject this head to protect the saved prefix.
        selected=0;accepted=False;replacement=None;updated=control_learner
        lower_response_applied=True
        # A solved no-edit head may itself break a previously solved grasp.
        # Reject that head rather than silently publishing an unsafe response.
        if self.fixed_start and not adapted_metrics[0]['saved_prefix_preserved']:
            updated=learner;lower_response_applied=False
        old_version=self.version
        if feasible:
            _,selected,candidate,updated=min(feasible,key=lambda item:item[0])
            lower_response_applied=True
            self.binding=candidate;self.version+=1;self.publish(step);accepted=True
            replacement=RLEnv(replace(self.setup.scene,robot_reference=self.binding),
                               self.setup.reference,config=deepcopy(env.env.config))
            assert_device_binding(replacement,self.binding)
            replacement.reset();replacement.capture_step();replacement.reset()
            replacement=WarpRLEnv(replacement,publish_diagnostics=False,publish_step_metrics=False)
        if command_digest(self.setup.reference)!=self.command_sha256:
            raise RuntimeError('response editor changed immutable object commands')
        evidence=self.output.parent/'search_batches'/f'step_{step}';evidence.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(evidence/'proposals.npz',actions=actions,joint_q=qs,joint_target=targets,
                            costs=costs,frozen_costs=frozen_costs)
        np.savez_compressed(evidence/'lower_responses.npz',kernel=responses.kernel,gain=responses.gain,
            bias=responses.bias,weights=responses.weights,
            teacher_steps=responses.teacher_steps,
            gradient_norm=responses.certificate.gradient_norm,
            parameter_error_bound=responses.certificate.parameter_error_bound,
            objective_gap_bound=responses.certificate.objective_gap_bound)
        save_snapshot(evidence/'selected_reference.npz',self.binding,self.projector.layout,
                      provenance=dict(step=step,trial=selected,accepted=accepted))
        record=dict(step=step,upper_round=self.upper_round,engine='response_bilevel',accepted=accepted,
            version_before=old_version,version_after=self.version,selected_trial=selected,
            candidate_worlds=self.upper_worlds,replicas_per_candidate=1,lower_policy_frozen=False,
            lower_backbone_critic_frozen_within_stage=True,lower_response_solved_for_every_candidate=True,
            lower_parameter_error_bound_max=float(responses.certificate.parameter_error_bound.max()),
            lower_objective_gap_bound_max=float(responses.certificate.objective_gap_bound.max()),
            lower_data_sha256=responses.data_sha256,selected_lower_trial=selected if lower_response_applied else None,
            lower_response_applied=lower_response_applied,
            baseline=adapted_metrics[0],
            baseline_cost_terms=self._cost_terms(adapted_metrics[0],qs[0]),
            selected_cost_terms=self._cost_terms(adapted_metrics[selected],qs[selected]),
            selected_lower_response_norm=float(np.linalg.norm(responses.weights[selected])),
            confirmation_attempts=attempts,upper_optimizer_updates=0,
            gate_diagnostics=dict(baseline_cost=float(costs[0]),noop_cost=float(costs[0]),
                improving_candidates=int(np.sum(costs<costs[0])),all_gates_pass=int(eligible.sum()),
                frozen_noop_cost=float(frozen_costs[0]),confirmed_control_cost=control_mean),
            candidates=[dict(trial=i,cost=float(costs[i]),frozen_cost=float(frozen_costs[i]),
                metrics=m,gate_passed=bool(eligible[i])) for i,m in enumerate(adapted_metrics)],
            failure_context=self.failure_context,windows=self.window_contexts,
            acceptance_mode=self.acceptance_mode, scoring_region=self.scoring_region,
            confirmation_reset_count=0 if self.local_acceptance else 3,
            seconds=time.perf_counter()-started,
            upper_simulation_transitions=self.validation_steps-previous_steps,
            cumulative_upper_simulation_transitions=self.validation_steps)
        if self.local_acceptance:
            record['local_cost_gain']=float(costs[0]-costs[selected])
            record['local_cost_gain_fraction']=float((costs[0]-costs[selected])/max(abs(costs[0]),1e-8))
            record['full_horizon_cost_before']=response_cost(adapted_metrics[0],qs[0],
                self.baseline.joint_q,self.projector.layout,self.baseline.fps,self.wrench_weight)
            record['full_horizon_cost_after']=response_cost(adapted_metrics[selected],qs[selected],
                self.baseline.joint_q,self.projector.layout,self.baseline.fps,self.wrench_weight)
            for row in record['candidates']:
                row['rejection_reasons']=self._local_rejections(row['metrics'],adapted_metrics[0])
                if row['cost']>=costs[0]-1e-4:
                    row['rejection_reasons'].append('no_local_cost_improvement')
            from collections import Counter
            record['gate_diagnostics']['rejection_counts']=dict(Counter(
                reason for row in record['candidates'][1:] for reason in row['rejection_reasons']))
        record.update(upper_learning)
        if self.fixed_start:
            if lower_response_applied:
                published_metrics=adapted_metrics[selected]
                p=candidate_feedback['position'][:,selected];a=candidate_feedback['orientation'][:,selected]
            else:
                published_metrics=baseline
                p=np.asarray(feedback['frame_per_object_position_m'])[:,0]
                a=np.asarray(feedback['frame_per_object_orientation_rad'])[:,0]
            progress=self._archive_pair(self.binding,updated,p,a,published_metrics,step,
                                       'accepted' if accepted else 'retained',save_pair=accepted)
            record.update(fixed_initial_state=True,initial_state_sha256=self.initial_state_sha256,
                          progress=progress,baseline_progress=baseline_progress,
                          failure_resolved_through_lookahead=bool(self.failure_context['failed'] and
                              progress['current_first_failure_seconds']>=self.scoring_region['end_seconds_exclusive']))
            np.savez_compressed(evidence/'nominal_baseline.npz',
                position_m=feedback['frame_per_object_position_m'],
                orientation_rad=feedback['frame_per_object_orientation_rad'])
            if not lower_response_applied:
                # The retained incoming actor did not have its wrench trace
                # rescored on the newly selected local interval.
                record['selected_cost_terms']=None
                record['published_full_horizon_cost']=progress['full_horizon_cost']
        record['upper_proposal'] = self.proposal_mode
        if self.proposal_learner:
            from flash_chord.training.flash_sac.checkpoint import save_learner_checkpoint
            save_learner_checkpoint(self.output.parent/'upper', self.proposal_learner.state,
                metadata={'lower_step':step, 'engine':'response_bilevel_upper_flashsac'}, save_replay=True)
        self.finalize_search_record(record,qs,costs,eligible)
        payload=json.dumps(record,allow_nan=False)+'\n'
        (evidence/'scores.json').write_text(payload)
        with (self.output.parent/'upper_updates.jsonl').open('a') as stream:stream.write(payload)
        print(f'[response-bilevel] accepted_reference={accepted} paired_lower_trial={selected} '+
              f'seconds={record["seconds"]:.1f}',flush=True)
        return ReferenceLearnerUpdate(updated,replacement)
