"""Replay a recorded proposal from a saved learner, without main-policy training.

Run inside a scheduler-owned GPU allocation. All outputs go to a new directory;
the source run, reference, checkpoint and archived demonstration are read only.
This validates finite-response selection and frozen confirmation with an actual
2048-world adaptation, rather than inferring acceptance from historical JSON.
The cost critic is restored from a PREVIOUS outer round and held fixed for
paired support-return diagnostics. Current-trial fitting cannot contaminate
these measurements. This does not test the online training schedule or certify
physical hypergradient accuracy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--critic-step', type=int,
                        help='Earlier upper-state checkpoint; defaults to the latest before --step')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    source, output = args.run_dir.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Validation output must be separate from the source run')
    checkpoint = source/'training'/f'state_{args.step}.safetensors'
    recorded_path = source/'training/search_batches'/f'step_{args.step}'/'scores.json'
    recorded = json.loads(recorded_path.read_text())
    old_candidate = next(candidate for candidate in recorded['candidates']
                         if candidate['window']['window_id'] == args.window)
    earlier = sorted(int(p.parent.name.removeprefix('step_')) for p in
        (source/'training/search_batches').glob('step_*/upper_state.msgpack')
        if int(p.parent.name.removeprefix('step_')) < args.step)
    critic_step = args.critic_step if args.critic_step is not None else (earlier[-1] if earlier else None)
    if critic_step is None or critic_step >= args.step:
        raise ValueError('Diagnostics require an upper critic from an earlier round to avoid trial leakage')
    saved_upper = source/'training/search_batches'/f'step_{critic_step}'/'upper_state.msgpack'
    critic_record = json.loads((saved_upper.parent/'scores.json').read_text())
    if critic_record['version_after'] != recorded['version_before']:
        raise ValueError('Prior-round critic and current learner use different reference versions')
    critic_digest = hashlib.sha256(saved_upper.read_bytes()).hexdigest()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if recorded['version_before'] != 0:
        raise ValueError('This diagnostic currently requires an unmodified source reference (version 0)')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output/'validation_input.json', dict(source_run=str(source),
        source_checkpoint=str(checkpoint), source_record=str(recorded_path),
        step=args.step, window=args.window, source_reference_version=recorded['version_before'],
        critic_source_step=critic_step, critic_checkpoint_sha256=critic_digest,
        source_rejection_reasons=old_candidate['rejection_reasons'],
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        purpose='Actual fixed-start finite-response diagnostic; no main-policy training'))

    import jax
    import jax.numpy as jnp
    import numpy as np
    import warp as wp
    from flax import serialization
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed, resolved_dict
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.checkpoint_metadata import (
        build_checkpoint_metadata, critic_schema, policy_schema, resolved_reference_config,
    )
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner
    from flash_chord.training.flash_sac.checkpoint import restore_learner_checkpoint
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor
    from flash_chord.retargeting.wrench_hypergradient import WrenchHypergradient, actor_policy_digest
    from flash_chord.retargeting.local_acceptance import scoring_region
    from flash_chord.retargeting.local_correction import local_envelope
    from flash_chord.retargeting.reference_relabel import binding_for_q
    from flash_chord.retargeting.lower_adaptation import install_learning
    from flash_chord.retargeting.response_editor import response_cost

    cfg = OmegaConf.load(source/'launch_resolved.yaml')
    cfg.training.device = args.device
    cfg.training.resume = False
    cfg.output_dir = str(output)
    cfg.logging.mode = 'disabled'
    OmegaConf.update(cfg, 'bilevel.adaptation.acceptance_mode', 'finite_response', force_add=True)
    OmegaConf.update(cfg, 'bilevel.diagnostic_save_trials', True, force_add=True)
    # This diagnostic replays a checkpoint; main-policy training is disabled.
    OmegaConf.update(cfg, 'bilevel.critic_online_interval_vector_steps', 0, force_add=True)
    OmegaConf.save(cfg, output/'validation_resolved.yaml')
    training = instantiate_typed(cfg.training, TrainingConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    if training.world_count != 2048 or cfg.scene.world_count != 2048:
        raise ValueError('This full-shape validation requires the recorded 2048-world configuration')
    seed = training.seed+args.step
    stage = replace(training.curriculum.stages[0], voc_scale=float(recorded['adaptation_voc_scale']))
    with wp.ScopedDevice(training.device):
        setup = setup_scene(parquet=cfg.task.parquet, control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed, embodiment=embodiment, collision=collision,
            world_count=cfg.scene.world_count, include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min, object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get('contact_friction', 1.),
            object_free_joint_damping=cfg.scene.get('object_free_joint_damping', 0.),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
            robot_reference_path=cfg.task.get('robot_reference_path'),
            robot_reference_sha256=cfg.task.get('robot_reference_sha256'))
        physical = RLEnv(setup.scene, setup.reference, config=env_config)
        physical.reset(); physical.capture_step(); physical.reset()
        env = WarpRLEnv(physical, publish_diagnostics=False, publish_step_metrics=False)
        env.set_curriculum(stage)
        environment_config = resolved_dict(cfg)
        environment_config['task']['parquet'] = str(setup.parquet_path)
        environment_config['reference'] = resolved_reference_config(setup.reference, str(setup.parquet_path))
        metadata = build_checkpoint_metadata(environment_config,
            policy_schema(physical.action, physical.observation_strategy),
            critic_schema(physical.action, physical.observation_strategy, env.critic_context_names))
        editor = WrenchHypergradientEditor(options=resolved_dict(cfg.bilevel), setup=setup,
            embodiment=embodiment, collision=collision, training=training,
            environment_config=environment_config, checkpoint_metadata=metadata, output_dir=output)
        device = resolve_jax_device(training.device)
        observation = jax.device_put(jnp.array(env.reset(), copy=True), device)
        context = jax.device_put(jnp.array(env.initial_critic_context, copy=True), device)
        editor.audit_initial_state(physical, 'training_start')
        learner, _ = create_learner(training, observation, context, action_dim=env.action_dim,
            objective_term_count=len(env.objective_term_names),
            termination_cause_count=len(env.termination_cause_names),
            tracking_error_count=len(env.tracking_error_names),
            critic_context_names=env.critic_context_names, device=device)
        learner, _ = restore_learner_checkpoint(checkpoint, learner, load_optimizer=True,
                                               load_reward_normalizer=True)
        if int(learner.environment_steps) != args.step:
            raise ValueError('Restored main training step differs from the recorded proposal')
        incoming_digest = actor_policy_digest(learner)
        archive = json.loads((source/'training/best_demo.json').read_text())
        errors_path = Path(archive['directory'])/'object_errors.npz'
        with np.load(errors_path) as errors:
            editor.demo_archive.position = errors['position_m'].copy()
            editor.demo_archive.orientation = errors['orientation_rad'].copy()
        editor.demo_archive.cost = float(archive['full_horizon_cost'])
        editor.demo_archive.best = deepcopy(archive)
        write_json(output/'historical_archive_source.json', archive)
        if archive['initial_state_sha256'] != editor.initial_state_sha256:
            raise ValueError('Historical archive and reproduced initial state differ')
        # Compute the upper observation shape using the real evaluator, then
        # restore the recorded detection/window rather than selecting new ones.
        baseline, _, _, feedback = editor.rollout(editor.binding, learner, seed, return_feedback=True)
        context_rows = editor.observation(baseline, feedback)
        editor.upper_round = int(recorded['upper_round'])
        editor.failure_context = deepcopy(recorded['failure_context'])
        editor.scoring_region = scoring_region(editor.failure_context, editor.binding.num_frames,
            editor.binding.fps, lookahead_seconds=float(editor.options.get('acceptance_lookahead_seconds', 1.)))
        editor.protected_prefix_frames = editor.scoring_region['start_frame']
        window = deepcopy(old_candidate['window'])
        editor.local_context = window
        editor.basis = local_envelope(editor.binding.num_frames, editor.binding.fps,
                                     window['start_seconds'], window['end_seconds'])[:, None]
        editor.before_search_q = editor.binding.joint_q.copy()
        editor.hyper = WrenchHypergradient(training, env.observation_dim,
            learner.critic_context.shape[-1], context_rows.shape[-1], editor.hg_config, seed)
        template = dict(actor=editor.hyper.upper_params, actor_optimizer=editor.hyper.upper_state,
            critic=editor.hyper.cost_params, target_critic=editor.hyper.target_params,
            critic_optimizer=editor.hyper.cost_state)
        restored = serialization.from_bytes(template, saved_upper.read_bytes())
        editor.hyper.upper_params, editor.hyper.upper_state = restored['actor'], restored['actor_optimizer']
        editor.hyper.cost_params, editor.hyper.target_params = restored['critic'], restored['target_critic']
        editor.hyper.cost_state = restored['critic_optimizer']
        baseline, _, _, _ = editor.rollout(editor.binding, learner, seed, return_feedback=True)
        baseline_cost = editor._cost(baseline, editor.binding.joint_q)
        theta = np.asarray(old_candidate['theta'])
        candidate = binding_for_q(editor, editor.candidate_joint_q(theta))
        active = np.flatnonzero(editor.basis[:, 0] > 0)
        editor._begin_proposal_audit(args.window, (int(active[0]), int(active[-1])+1), args.step)
        budgets = editor.adaptation_config.budgets(env.world_count, editor.binding.num_frames,
                                                  region=editor.scoring_region)
        if list(budgets) != recorded['adaptation_checkpoints']:
            raise ValueError('Diagnostic budgets differ from the historical matched-training budgets')
        changes = {}
        for hand in editor.projector.layout.hands:
            fingers = hand.finger_q_ids
            wrist = [qid for name, qid in zip(editor.projector.layout.scalar_joints.names,
                                             editor.projector.qids)
                     if name.startswith(hand.side+'_') and name.endswith(('wrist_x', 'wrist_y', 'wrist_z'))]
            if len(wrist) != 3:
                raise ValueError(f'Expected three named wrist translation joints for {hand.side}')
            delta = candidate.joint_q-editor.binding.joint_q
            changes[hand.side+'_wrist_peak_cm'] = float(np.linalg.norm(delta[:, wrist], axis=-1).max()*100)
            changes[hand.side+'_finger_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean(delta[:, fingers]**2))))
            changes[hand.side+'_finger_peak_deg'] = float(np.rad2deg(np.abs(delta[:, fingers]).max()))
        print('[finite-response-validation] restored main step', args.step,
              'budgets', budgets, 'reference change', changes, flush=True)
        controls = {}
        for result in editor._adaptation_branch(editor.binding, learner, env, stage,
                                                seed, budgets, 'noedit', args.step):
            controls[result['environment_steps']] = {key: result[key]
                for key in ('cost', 'metrics', 'training', 'snapshot')}
        result, control, adaptation = editor._matched_adaptation(candidate, learner, env, stage,
            seed, budgets, controls, f'window_{args.window}', args.step,
            baseline=baseline, baseline_cost=baseline_cost)
        confirmation = dict(attempted=False, passed=False, environment_steps=result['environment_steps'])
        selected_metrics, selected_feedback = result['metrics'], result['feedback']
        if adaptation['physical_passed_at_selected_checkpoint']:
            confirmation, selected_metrics, selected_feedback = editor._confirm_frozen_pair(
                candidate, result, control, learner, seed, baseline, baseline_cost)
        accepted = bool(confirmation['passed'])
        selected_learner = install_learning(learner, result['snapshot'])
        p = np.asarray(selected_feedback['frame_per_object_position_m'])[:, 0]
        a = np.asarray(selected_feedback['frame_per_object_orientation_rad'])[:, 0]
        origin = 'confirmed_validation' if accepted else 'rejected_validation'
        full_horizon_cost = response_cost(selected_metrics, candidate.joint_q,
            editor.baseline.joint_q, editor.projector.layout, editor.baseline.fps,
            editor.wrench_weight, local=False)
        pair = editor.demo_archive._save_pair(candidate, selected_learner, p, a,
            selected_metrics, full_horizon_cost, args.step,
            origin, editor.demo_archive.check(p, a))
        if actor_policy_digest(learner) != incoming_digest:
            raise RuntimeError('Validation mutated the incoming controller')
        report = dict(source_run=str(source), source_checkpoint=str(checkpoint),
            critic_source_step=critic_step, critic_checkpoint_sha256=critic_digest,
            critic_diagnostic='prior-round frozen critic, matched physical frames, policy-specific observed support returns',
            online_critic_schedule_tested=False, physical_hypergradient_tested=False,
            step=args.step, window=window, accepted=accepted,
            status='confirmed_finite_response' if accepted else 'rejected_diagnostic_only',
            main_training_steps=0, adaptation_transitions=editor.adaptation_transitions,
            upper_simulation_transitions=editor.validation_steps,
            incoming_policy_sha256=incoming_digest, initial_state_sha256=editor.initial_state_sha256,
            source_rejection_reasons=old_candidate['rejection_reasons'],
            reference_change=changes, proposed_theta=theta.tolist(),
            scoring_region=editor.scoring_region, adaptation_checkpoints=list(budgets),
            baseline_cost=baseline_cost, baseline_metrics=baseline,
            candidate_cost=result['cost'], control_cost=control['cost'],
            adaptation=adaptation, confirmation=confirmation,
            selected_pair=pair, critic_feedback=editor._round_critic_feedback,
            seconds=time.perf_counter()-started)
        write_json(output/'validation_report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'accepted', 'candidate_cost',
            'control_cost', 'baseline_cost', 'reference_change', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
