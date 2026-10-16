"""Portable recipe for measured critic feedback and bounded FlashSAC validation."""
import json


def configure_hypergradient(command, config, phase, worlds, base):
    """Replace the old candidate-search recipe without changing physics or media."""
    smoke = phase == 'smoke'
    # Smoke exercises the same 2,048-world geometry, replay and batch size as
    # production, but stops after two real upper rounds and checkpoint media.
    total = 1048576 if smoke else 100663296
    interval = 501760
    checkpoint = 1048576 if smoke else 5017600
    replace_keys = {
        'training.total_environment_steps', 'training.replay.capacity',
        'training.replay.minimum_size', 'training.checkpoint.save_interval_environment_steps',
        '+adaptive_voc.interval',
        'scene.support',
        'scene.object_initial_pose_overrides',
    }
    command[:] = [arg for arg in command
                  if not arg.startswith(('+bilevel.', 'bilevel.', '+bilevel='))
                  and arg.split('=', 1)[0] not in replace_keys]
    limits = config['episode'].get('controller_limits', {})
    scope = config.get('upper_edit_scope', 'full')
    if scope not in ('full', 'wrists_only'):
        raise ValueError(f'Unsupported upper edit scope: {scope}')
    upper_dimension = 12 if scope == 'wrists_only' else 56
    command.extend([
        '+bilevel=wrench_hypergradient',
        f'bilevel.edit_scope={scope}',
        f'scene.support={str(bool(config["episode"].get("support_asset"))).lower()}',
        'action.wrist_pos_clip=0.05',
        'action.wrist_ori_clip=0.5',
        'observation.object_body_velocity.enabled=true',
        'action.wrist_residual_linear_speed_limit_m_s=0.15',
        'action.wrist_residual_angular_speed_limit_rad_s=1.5',
        f"+env.wrist_linear_speed_limit_m_s={limits.get('linear_speed_limit_m_s', 1.0)}",
        f"+env.wrist_angular_speed_limit_rad_s={limits.get('angular_speed_limit_rad_s', 12.0)}",
        'bilevel.hypergradient.inner_optimizer=adam',
        'bilevel.hypergradient.cost_scale=0.01',
        'bilevel.hypergradient.trial_mc_weight=1.0',
        'bilevel.critic_online_interval_vector_steps=16',
        'bilevel.critic_online_updates=4',
        'bilevel.diagnostic_save_trials=true',
        f'bilevel.warmup={interval}', f'bilevel.interval={interval}',
        f'training.total_environment_steps={total}',
        'training.replay.capacity=1048576',
        'training.replay.minimum_size=102400',
        f'training.checkpoint.save_interval_environment_steps={checkpoint}',
        f'+adaptive_voc.interval={interval}',
        'bilevel.adaptation.budget_mode=bounded_window',
        'bilevel.adaptation.acceptance_mode=finite_response',
        'bilevel.adaptation.minimum_vector_steps=64',
        'bilevel.adaptation.checkpoint_stride_vector_steps=32',
        'bilevel.adaptation.maximum_vector_steps=448',
        'bilevel.adaptation.window_tail_vector_steps=16',
    ])
    ground_plane = config['episode'].get('ground_plane')
    if ground_plane is not None:
        command.append('+collision.ground_plane=[' + ','.join(map(str, ground_plane)) + ']')
    initial_poses = config['episode'].get('object_initial_pose_overrides')
    if initial_poses is not None:
        if config['episode'].get('initial_state_scope') != 'frame_zero_only':
            raise ValueError('Initial object pose overrides require explicit frame-zero-only provenance')
        entries = [name + ':{position:' + json.dumps(pose['position']) +
                   ',quaternion_wxyz:' + json.dumps(pose['quaternion_wxyz']) + '}'
                   for name, pose in initial_poses.items()]
        command.append('scene.object_initial_pose_overrides={' + ','.join(entries) + '}')
        config['object_initial_pose_overrides'] = initial_poses
        config['initial_state_report_sha256'] = config['episode']['initial_state_report_sha256']
        config['initial_state_scope'] = 'frame_zero_only'
        config['initialization_reference_targets_modified'] = False
        config['training_reset'] = 'frame-zero robot reference, validated object initial poses, and zero velocities'
    config['wrist_control'].update(
        linear_speed_limit_m_s=limits.get('linear_speed_limit_m_s', 1.0),
        angular_speed_limit_rad_s=limits.get('angular_speed_limit_rad_s', 12.0))
    # Do not inherit convex-response guarantees or population-search labels.
    for obsolete in ('upper_trials', 'replicas_per_candidate', 'lower_surrogate',
                     'lower_response_error_tolerance', 'guarantee_document'):
        config.pop(obsolete, None)
    config.update(
        engine='wrench_hypergradient', upper_optimizer='wrench_hypergradient',
        ablation_variant='wrench_hypergradient_online_critic_finite_response',
        experiment_name='wrench-stable-control', experiment_date='20261016',
        upper_edit_scope=scope,
        total_environment_steps=total, upper_search_interval=interval,
        voc_probe_interval=interval,
        upper_search_warmup=interval, checkpoint_interval=checkpoint,
        planned_upper_search_rounds=(total-1)//interval,
        training_command=command, worlds=worlds, eval_worlds=1,
        adaptation_worlds=worlds, nominal_candidates_per_window=1,
        maximum_proposals_per_round=4, upper_action_dimension=upper_dimension,
        upper_updates_per_batch=4,
        upper_proposal_method='one bounded actor proposal per detected window; finite-step response hypergradient',
        upper_feedback='discounted CHORD-style wrench-support cost; not net six-axis wrench',
        proposal_response='3 projected Adam actor updates from incoming moments and schedule; frozen lower critics and statistics',
        proposal_inner_optimizer='adam', upper_cost_scale=0.01,
        critic_online_interval_vector_steps=16, critic_online_updates=4,
        diagnostic_save_trials=True,
        actor_observes_object_velocity=True,
        lower_control=dict(wrist_position_residual_clip_m=.05, wrist_orientation_residual_clip_rad=.5,
            wrist_residual_linear_speed_limit_m_s=.15, wrist_residual_angular_speed_limit_rad_s=1.5,
            angular_residual_coordinates='RPY; coordinate-rate bound, not physical angular speed',
            reference_motion_slew_limited=False, physical_speed_limits=config['wrist_control']),
        trial_mc_weight=1.0, outer_validation='current-round unassisted baseline only',
        lower_response='matched full FlashSAC training on fresh data for candidate and unchanged control',
        adaptation_budget_mode='bounded_window',
        adaptation_acceptance_mode='finite_response',
        adaptation_minimum_vector_steps=64,
        adaptation_checkpoint_stride_vector_steps=32,
        adaptation_maximum_vector_steps=448,
        adaptation_window_tail_vector_steps=16,
        adaptation_maximum_transitions_per_branch=448*worlds,
        adaptation_budget_rule='three checkpoints covering the common scoring interval where possible; strict 448-vector-step cap',
        measured_critic_feedback='complete measured unassisted baseline and candidate/control returns, including rejected proposals; branch reference and policy kept separate',
        adaptation_clock='separate from main environment steps; main training pauses during validation',
        acceptance_score='best admissible matched finite-training checkpoint; local pose/wrench improvement, per-object and archived-prefix guards; frozen same-initial-state confirmation',
        adaptation_stability='diagnostic only; convergence is not required for the finite-response objective',
        guarantee_scope='gradient of short finite actor response and learned cost surrogate; long training is validation, not differentiated',
        method_document=str(base/'flash_chord/docs/wrench_hypergradient.txt'),
        experiment_interpretation='Finite-step bilevel gradient proposals with longer full FlashSAC validation; no global convergence claim',
    )
