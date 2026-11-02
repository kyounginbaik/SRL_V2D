"""Warm-start a changed robot reference without pretending to resume a run.

Network weights and reward statistics transfer; replay, optimizer state, clocks,
environment state, exploration and random keys belong to the new experiment.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def adaptation_configuration(configuration, worlds, steps, interval):
    """Preserve training physics/reward while removing assistance and old clocks."""
    if min(worlds, steps, interval) <= 0 or steps % worlds or interval % worlds:
        raise ValueError('positive adaptation budgets divisible by world count required')
    config = deepcopy(configuration)
    config['scene']['world_count'] = worlds
    config['env'].update(auto_reset=True, voc_scale=0.)
    config['env']['voc'].update(max_force=0., max_torque=0.)
    config['env']['reset'].update(always_reset_to_first_frame=True, reset_finger_openness=1., voc_decay_steps=0)
    training = config['training']
    training.update(world_count=worlds, total_environment_steps=steps, resume=False, checkpoint_path=None)
    curriculum = training['curriculum']
    curriculum['stages'] = [curriculum['stages'][0]]  # same reward used by the step-VOC recipe
    curriculum['thresholds'] = [0]
    curriculum['stages'][0].update(voc_scale=0., reset_to_first_frame_probability=1.)
    training['checkpoint']['save_interval_environment_steps'] = interval
    return config


def transfer_initialization(fresh, restored):
    return fresh.replace(
        actor=fresh.actor.replace(params=restored.actor.params, batch_stats=restored.actor.batch_stats),
        critic=fresh.critic.replace(params=restored.critic.params, batch_stats=restored.critic.batch_stats),
        target_critic=restored.target_critic,
        temperature=fresh.temperature.replace(log_temperature=restored.temperature.log_temperature),
        reward_normalizer=restored.reward_normalizer.replace(
            discounted_return=fresh.reward_normalizer.discounted_return),
    )


def restore_initialization(state_path, fresh):
    """Load learned values while retaining transient buffers BEFORE shape checks.

    A normal resume restore requires the old world count. Warm-starting instead
    preserves the new world's reward-return/noise buffers, optimizers and clocks.
    Learned parameter shapes and dtypes still undergo strict checkpoint checks.
    """
    import jax
    from flash_chord.training.checkpoint import _restore_checkpoint, read_checkpoint_metadata
    from flash_chord.training.flash_sac.checkpoint import training_checkpoint
    metadata = read_checkpoint_metadata(state_path)
    if metadata.get('algorithm') != 'flash_sac' or metadata.get('checkpoint_kind') != 'training_state':
        raise ValueError('warm start requires a FlashSAC training-state checkpoint')
    template = training_checkpoint(fresh)
    learned = ('.actor.params', '.actor.batch_stats', '.critic.params', '.critic.batch_stats',
               '.target_critic', '.temperature.log_temperature', '.reward_normalizer.moments')
    def should_load(path):
        return any(path == prefix or path.startswith(prefix + '.') or path.startswith(prefix + '[')
                   for prefix in learned)
    retained = frozenset(jax.tree_util.keystr(path) for path, _ in
                         jax.tree_util.tree_flatten_with_path(template)[0]
                         if not should_load(jax.tree_util.keystr(path)))
    restored, _ = _restore_checkpoint(state_path, template, template_leaf_paths=retained)
    return transfer_initialization(fresh, restored)


@contextmanager
def warm_start_runner(state_path, current_metadata, output_dir):
    """Install a one-shot initializer around the existing FlashSAC runner.

    Schema validation happens before weights are transferred. A separate new
    training run may intentionally use a changed robot reference; this is not
    supported as an exact bilevel resume and no old upper state is restored.
    """
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    state_path = Path(state_path)
    saved = read_checkpoint_metadata(state_path)
    schemas = runner._required_checkpoint_schemas(current_metadata)
    runner._validate_resume_schemas(saved, *schemas)
    original = runner.create_learner
    initialized = False

    def initialize(*args, **kwargs):
        nonlocal initialized
        fresh, executables = original(*args, **kwargs)
        if initialized:
            raise RuntimeError('warm initializer must only initialize the lower learner once')
        learner = restore_initialization(state_path, fresh)
        initialized = True
        manifest = dict(source=str(state_path), sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                        transferred=['actor weights', 'critic weights', 'target critic', 'temperature', 'reward statistics'],
                        fresh=['replay', 'optimizers', 'environment steps', 'update clocks', 'environment state', 'random keys', 'exploration'])
        (Path(output_dir) / 'warm_start.json').write_text(json.dumps(manifest, indent=2))
        print('[warm-start] weights restored; replay, optimizers and experiment clocks are fresh', flush=True)
        return learner, executables

    runner.create_learner = initialize
    try:
        yield
    finally:
        runner.create_learner = original
