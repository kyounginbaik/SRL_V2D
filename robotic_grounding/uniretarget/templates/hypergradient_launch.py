"""Portable recipe for measured critic feedback and bounded FlashSAC validation."""


def configure_hypergradient(command, config, phase, worlds, base):
    """Replace the old candidate-search recipe without changing physics or media."""
    smoke = phase == 'smoke'
    # Smoke exercises the same 2,048-world geometry, replay and batch size as
    # production, but stops after one real upper round and checkpoint media.
    total = 524288 if smoke else 100663296
    interval = 501760
    checkpoint = 524288 if smoke else 1003520
    replace_keys = {
        'training.total_environment_steps', 'training.replay.capacity',
        'training.replay.minimum_size', 'training.checkpoint.save_interval_environment_steps',
        '+adaptive_voc.interval',
    }
    command[:] = [arg for arg in command
                  if not arg.startswith(('+bilevel.', 'bilevel.', '+bilevel='))
                  and arg.split('=', 1)[0] not in replace_keys]
    command.extend([
        '+bilevel=wrench_hypergradient',
        f'bilevel.warmup={interval}', f'bilevel.interval={interval}',
        f'training.total_environment_steps={total}',
        'training.replay.capacity=1048576',
        'training.replay.minimum_size=102400',
        f'training.checkpoint.save_interval_environment_steps={checkpoint}',
        f'+adaptive_voc.interval={interval}',
        'bilevel.adaptation.budget_mode=bounded_window',
        'bilevel.adaptation.minimum_vector_steps=64',
        'bilevel.adaptation.checkpoint_stride_vector_steps=32',
        'bilevel.adaptation.maximum_vector_steps=448',
        'bilevel.adaptation.window_tail_vector_steps=16',
    ])
    # Do not inherit convex-response guarantees or population-search labels.
    for obsolete in ('upper_trials', 'replicas_per_candidate', 'lower_surrogate',
                     'lower_response_error_tolerance', 'guarantee_document'):
        config.pop(obsolete, None)
    config.update(
        engine='wrench_hypergradient', upper_optimizer='wrench_hypergradient',
        ablation_variant='wrench_hypergradient_measured_feedback_bounded_sac',
        experiment_name='wrench-feedback-fast', experiment_date='20261016',
        total_environment_steps=total, upper_search_interval=interval,
        voc_probe_interval=interval,
        upper_search_warmup=interval, checkpoint_interval=checkpoint,
        planned_upper_search_rounds=(total-1)//interval,
        training_command=command, worlds=worlds, eval_worlds=1,
        adaptation_worlds=worlds, nominal_candidates_per_window=1,
        maximum_proposals_per_round=4, upper_action_dimension=56,
        upper_updates_per_batch=4,
        upper_proposal_method='one bounded actor proposal per detected window; finite-step response hypergradient',
        upper_feedback='discounted CHORD-style wrench-support cost; not net six-axis wrench',
        proposal_response='3 projected SGD actor updates on replay; frozen lower critics and statistics',
        lower_response='matched full FlashSAC training on fresh data for candidate and unchanged control',
        adaptation_budget_mode='bounded_window',
        adaptation_minimum_vector_steps=64,
        adaptation_checkpoint_stride_vector_steps=32,
        adaptation_maximum_vector_steps=448,
        adaptation_window_tail_vector_steps=16,
        adaptation_maximum_transitions_per_branch=448*worlds,
        adaptation_budget_rule='three checkpoints covering the common scoring interval where possible; strict 448-vector-step cap',
        measured_critic_feedback='completed unassisted candidate/control rollout segments, including rejected proposals; branch reference and policy kept separate',
        adaptation_clock='separate from main environment steps; main training pauses during validation',
        acceptance_score='stable matched-training comparison, local pose/wrench improvement, per-object and archived-prefix guards',
        guarantee_scope='gradient of short finite actor response and learned cost surrogate; long training is validation, not differentiated',
        method_document=str(base/'flash_chord/docs/wrench_hypergradient.txt'),
        experiment_interpretation='Finite-step bilevel gradient proposals with longer full FlashSAC validation; no global convergence claim',
    )
