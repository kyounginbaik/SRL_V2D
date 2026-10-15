"""Fresh-experience FlashSAC adaptation for matched reference comparisons.

The short differentiable actor response proposes an edit. This separate path
trains the complete lower learner before accepting it. Trial work has its own
transition counter; it never consumes or donates the incoming learner's arrays.
"""
from dataclasses import dataclass, replace
import time

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.training.flash_sac.learner import create_learner, Transition
from flash_chord.training.flash_sac.replay import create_replay, ReplaySpec
from flash_chord.training.flash_sac.exploration import create_exploration
from flash_chord.training.flash_sac.runner import resolve_jax_device


@dataclass(frozen=True)
class AdaptationConfig:
    checkpoints: tuple[int, ...] = (50_000, 100_000, 200_000)
    minimum_reference_passes: tuple[int, ...] = (1, 2, 4)
    replay_capacity: int = 262_144
    replay_minimum_size: int = 8192
    minimum_actor_updates: int = 32
    minimum_critic_updates: int = 32
    minimum_window_samples: int = 128
    stable_comparisons: int = 1
    cost_relative_tolerance: float = .05
    position_tolerance_m: float = .003
    orientation_tolerance_rad: float = .03
    success_tolerance: float = .02
    budget_mode: str = 'reference_passes'
    minimum_vector_steps: int = 64
    checkpoint_stride_vector_steps: int = 32
    maximum_vector_steps: int = 256
    window_tail_vector_steps: int = 16

    def __post_init__(self):
        counts = (*self.checkpoints, *self.minimum_reference_passes, self.replay_capacity,
                  self.replay_minimum_size, self.minimum_actor_updates,
                  self.minimum_critic_updates, self.minimum_window_samples, self.stable_comparisons,
                  self.minimum_vector_steps, self.checkpoint_stride_vector_steps,
                  self.maximum_vector_steps, self.window_tail_vector_steps)
        if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in counts):
            raise ValueError('positive integer adaptation counts required')
        if (len(self.checkpoints) < 2 or len(self.checkpoints) != len(self.minimum_reference_passes)
                or any(b <= a for a, b in zip(self.checkpoints, self.checkpoints[1:]))
                or any(b <= a for a, b in zip(self.minimum_reference_passes, self.minimum_reference_passes[1:]))
                or self.replay_minimum_size > self.replay_capacity
                or self.stable_comparisons >= len(self.checkpoints)):
            raise ValueError('ordered paired adaptation checkpoints and valid replay limits required')
        tolerances = (self.cost_relative_tolerance, self.position_tolerance_m,
                      self.orientation_tolerance_rad, self.success_tolerance)
        if any(not np.isfinite(v) or v <= 0 for v in tolerances):
            raise ValueError('finite positive adaptation stability tolerances required')
        if self.budget_mode not in ('reference_passes', 'bounded_window'):
            raise ValueError('adaptation budget_mode must be reference_passes or bounded_window')
        if (self.budget_mode == 'bounded_window' and self.maximum_vector_steps <
                self.minimum_vector_steps+(len(self.checkpoints)-1)*self.checkpoint_stride_vector_steps):
            raise ValueError('bounded adaptation must fit all distinct checkpoints inside its hard cap')

    def budgets(self, worlds, frames, *, region=None):
        """Aggregate transitions for equally budgeted candidate/control branches.

        Legacy reference-pass budgets span whole demonstrations. Bounded-window
        budgets instead allow one visit to the shared scoring interval plus a
        short tail, followed by nearby checkpoints. Neither schedule guarantees
        coverage: failed worlds can reset before the interval. Readiness always
        checks observed frame samples, and the bounded mode never silently grows
        its vector-step cap to reach a late interval.
        """
        if (isinstance(worlds, bool) or isinstance(frames, bool) or
                int(worlds) != worlds or int(frames) != frames or worlds < 1 or frames < 2):
            raise ValueError('positive world count and at least two reference frames required')
        worlds, frames = int(worlds), int(frames)
        if self.budget_mode == 'bounded_window':
            stop = self._region_stop(region, frames)
            warmup = (self.replay_minimum_size+worlds-1)//worlds
            desired_first = max(self.minimum_vector_steps, stop+self.window_tail_vector_steps,
                                warmup+self.window_tail_vector_steps)
            last_first = self.maximum_vector_steps-(len(self.checkpoints)-1)*self.checkpoint_stride_vector_steps
            first = min(desired_first, last_first)
            if first <= warmup:
                raise ValueError('bounded adaptation cap cannot fit replay warmup before the first checkpoint')
            return tuple((first+i*self.checkpoint_stride_vector_steps)*worlds
                         for i in range(len(self.checkpoints)))
        return tuple(int(np.ceil(max(request, passes*(frames-1)*worlds)/worlds))*worlds
                     for request, passes in zip(self.checkpoints, self.minimum_reference_passes))

    @staticmethod
    def _region_stop(region, frames):
        if region is None:
            raise ValueError('bounded-window adaptation requires the shared scoring region')
        start, stop = region['start_frame'], region['stop_frame_exclusive']
        if (isinstance(start, bool) or isinstance(stop, bool) or int(start) != start or int(stop) != stop
                or not 0 <= start < stop <= frames):
            raise ValueError('adaptation scoring region must be a nonempty interval inside the reference')
        return int(stop)

    def budget_report(self, worlds, frames, *, region=None):
        """Expose the cap and nominal coverage without claiming actual readiness."""
        budgets = self.budgets(worlds, frames, region=region)
        vectors = tuple(b//worlds for b in budgets)
        report = dict(mode=self.budget_mode, environment_steps=list(budgets),
                      vector_steps=list(vectors), world_count=int(worlds))
        if self.budget_mode == 'bounded_window':
            stop = self._region_stop(region, frames)
            report.update(maximum_vector_steps=self.maximum_vector_steps,
                scoring_stop_frame_exclusive=stop,
                required_vector_steps_with_tail=stop+self.window_tail_vector_steps,
                first_checkpoint_can_reach_scoring_end=vectors[0] >= stop,
                final_checkpoint_can_reach_scoring_end=vectors[-1] >= stop,
                final_checkpoint_includes_requested_tail=vectors[-1] >= stop+self.window_tail_vector_steps,
                cap_reached=vectors[-1] == self.maximum_vector_steps)
        return report


LEARNING_FIELDS = ('actor', 'critic', 'target_critic', 'temperature',
                   'reward_normalizer', 'loss_scale', 'global_update_step')


def learning_snapshot(state):
    """Compact independent host snapshot: no replay or live simulator buffers."""
    return {name: jax.tree.map(lambda x: np.array(x, copy=True), getattr(state, name))
            for name in LEARNING_FIELDS}


def install_learning(state, snapshot, device=None):
    """Publish all tested learning state, preserving the main training clock."""
    values = jax.tree.map(lambda x: jax.device_put(np.array(x, copy=True), device), snapshot)
    # A fresh reset must not inherit an unfinished trajectory's return sum.
    values['reward_normalizer'] = values['reward_normalizer'].replace(
        discounted_return=jnp.zeros_like(values['reward_normalizer'].discounted_return))
    return state.replace(**values)


class FlashSACAdaptation:
    """One continuing trial with fresh replay and the normal donated SAC update."""
    def __init__(self, env, training, settings, initial_snapshot, stage, seed, *, executables=None):
        self.env, self.settings = env, settings
        self.device = resolve_jax_device(training.device)
        minimum = max(settings.replay_minimum_size, training.replay.batch_size)
        if settings.replay_capacity < max(minimum, env.world_count):
            raise ValueError('adaptation replay must fit a collection and the lower training batch')
        self.config = replace(training, resume=False, checkpoint_path=None,
            optimization=replace(training.optimization,
                learning_rate_schedule_updates=training.learning_rate_schedule_updates),
            replay=replace(training.replay, capacity=settings.replay_capacity,
                minimum_size=minimum, recent_window=None),
            exploration=replace(training.exploration, warmup_action='policy'))
        self.stage = stage
        env.set_curriculum(stage)
        observation = jax.device_put(np.array(env.reset(), copy=True), self.device)
        context = jax.device_put(np.array(env.initial_critic_context, copy=True), self.device)
        if executables is None:
            state, self.executables = create_learner(self.config, observation, context,
                action_dim=env.action_dim, objective_term_count=len(env.objective_term_names),
                termination_cause_count=len(env.termination_cause_names),
                tracking_error_count=len(env.tracking_error_names),
                critic_context_names=tuple(env.critic_context_names), device=self.device)
        else:
            # A small fresh template is supplied by the caller; never copy the
            # main run's million-row replay or recreate the compiled networks.
            template, self.executables = executables
            spec = ReplaySpec.from_config(self.config.replay, discount=self.config.discount,
                world_count=env.world_count, observation_dim=env.observation_dim,
                action_dim=env.action_dim, objective_term_count=len(env.objective_term_names),
                critic_context_dim=context.shape[-1])
            state = template.replace(replay=create_replay(spec, device=self.device),
                logging=jax.tree.map(lambda x: jax.device_put(np.zeros_like(x), self.device), template.logging),
                observation=observation, critic_context=context,
                exploration=create_exploration(env.world_count, env.action_dim, self.device),
                update_credit=jnp.asarray(0., jnp.float32), environment_steps=jnp.asarray(0, jnp.int32))
        self.state = install_learning(state, initial_snapshot, self.device).replace(key=jax.random.PRNGKey(seed))
        self.start_actor_updates = int(self.state.actor.schedule_step)
        self.start_critic_updates = int(self.state.critic.schedule_step)
        self.initial_snapshot = initial_snapshot
        self.steps = 0
        self.frames = np.zeros(1, np.int64)
        self.weights = jnp.asarray(stage.weights_for(env.objective_term_names), jnp.float32)
        self.frame_dt = jnp.asarray(env.frame_dt, jnp.float32)
        self.voc = jnp.asarray(stage.voc_scale, jnp.float32)
        self.false = jnp.asarray(False)
        self.action, exploration, key = self.executables.initial_action(
            self.state.actor.params, self.state.actor.batch_stats, self.state.replay.size,
            self.state.observation, self.state.exploration, self.state.key)
        self.state = self.state.replace(exploration=exploration, key=key)
        self.last_progress = time.monotonic()

    def advance(self, total_steps, frame_reader):
        if total_steps < self.steps or total_steps % self.env.world_count:
            raise ValueError('adaptation target must advance by complete vector collections')
        while self.steps < total_steps:
            frames = np.asarray(frame_reader(), dtype=int)
            if np.any(frames < 0):
                raise ValueError('adaptation encountered a negative reference frame')
            counts = np.bincount(frames)
            if len(counts) > len(self.frames):
                self.frames = np.pad(self.frames, (0, len(counts)-len(self.frames)))
            self.frames[:len(counts)] += counts
            transition = self.env.step(self.action)
            if any(x is None for x in (transition.objective_terms, transition.terminal_observation,
                                       transition.critic_context, transition.terminal_critic_context)):
                raise ValueError('adaptation needs terminal-state observations and raw objective terms')
            progress, diagnostics = self.env.transition_logging_inputs()
            self.state, self.action, metrics = self.executables.consume_transition(self.state, Transition(
                action=self.action, objective_terms=transition.objective_terms,
                terminated=transition.terminated, truncated=transition.truncated,
                episode_reference_progress=progress, termination_diagnostics=diagnostics,
                next_observation=transition.terminal_observation,
                next_critic_context=transition.terminal_critic_context,
                post_reset_observation=transition.observation, post_reset_critic_context=transition.critic_context),
                self.weights, self.frame_dt, self.voc, self.false, self.false)
            self.steps += self.env.world_count
            if time.monotonic()-self.last_progress > 30:
                print(f'[lower-adaptation] transitions={self.steps}/{total_steps}', flush=True)
                self.last_progress = time.monotonic()
        jax.block_until_ready(self.state.actor.params)
        if not all(np.isfinite(np.asarray(v)).all() for name in LEARNING_FIELDS
                   for v in jax.tree.leaves(getattr(self.state, name))):
            raise FloatingPointError('nonfinite lower adaptation state')
        return self.report()

    def report(self):
        actor_updates = int(self.state.actor.schedule_step)-self.start_actor_updates
        critic_updates = int(self.state.critic.schedule_step)-self.start_critic_updates
        changes = {}
        for name in ('actor', 'critic'):
            before = self.initial_snapshot[name].params
            after = getattr(self.state, name).params
            changes[name+'_parameter_delta_l2'] = float(np.sqrt(sum(
                np.sum((np.asarray(a)-b)**2, dtype=np.float64)
                for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(before)))))
        return dict(environment_steps=self.steps, actor_updates=actor_updates, critic_updates=critic_updates,
            replay_rows=int(self.state.replay.size), max_reference_frame=len(self.frames)-1,
            frame_samples=self.frames.tolist(), voc_scale=float(self.stage.voc_scale), **changes)

    def cache_template(self):
        """Retain shape-only host arrays; do not pin a trial replay on the GPU."""
        return (self.state.replace(replay=None,
            **{field: None for field in LEARNING_FIELDS},
            logging=jax.tree.map(lambda x: np.zeros(x.shape, x.dtype), self.state.logging),
            observation=None, critic_context=None, exploration=None, key=None,
            update_credit=None, environment_steps=None), self.executables)


def training_readiness(report, settings, region, *, require_coverage=True):
    reasons = []
    if report['actor_updates'] < settings.minimum_actor_updates:
        reasons.append('insufficient_lower_actor_updates')
    if report['critic_updates'] < settings.minimum_critic_updates:
        reasons.append('insufficient_lower_critic_updates')
    if report['actor_parameter_delta_l2'] <= 0 or report['critic_parameter_delta_l2'] <= 0:
        reasons.append('lower_parameters_did_not_change')
    counts = report['frame_samples']
    lo, hi = region['start_frame'], region['stop_frame_exclusive']
    if require_coverage and (sum(counts[lo:hi]) < settings.minimum_window_samples or report['max_reference_frame'] < hi-1):
        reasons.append('insufficient_failure_region_experience')
    return reasons


def comparison_stable(previous, current, settings):
    """Empirical plateau of both branches AND their advantage; not convergence."""
    return not comparison_instability_reasons(previous, current, settings)


def comparison_instability_reasons(previous, current, settings):
    """Plateau checks, with success fractions resolved to one scored frame."""
    def close(a, b, absolute=None):
        a, b = np.asarray(a), np.asarray(b)
        tolerance = absolute if absolute is not None else settings.cost_relative_tolerance*np.maximum(1., np.maximum(abs(a), abs(b)))
        return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                    and np.all(abs(a-b) <= tolerance))
    reasons = []
    for side in ('candidate', 'control'):
        old, new = previous[side], current[side]
        success_tolerance = settings.success_tolerance
        interval_keys = ('local_start_frame', 'local_stop_frame_exclusive')
        if all(key in item['metrics'] for item in (old, new) for key in interval_keys):
            intervals = [tuple(item['metrics'][key] for key in interval_keys) for item in (old, new)]
            valid = all(all(isinstance(value, (int, float, np.integer, np.floating))
                            and not isinstance(value, (bool, np.bool_))
                            and np.isfinite(value) and int(value) == value for value in interval)
                        and 0 <= interval[0] < interval[1] for interval in intervals)
            if not valid:
                reasons.append(side+'_local_scoring_interval_invalid')
            elif intervals[0] != intervals[1]:
                reasons.append(side+'_local_scoring_interval_changed')
            else:
                # At N scored frames, one changed success label moves the
                # fraction by 1/N. A smaller tolerance requires exact agreement.
                count = intervals[0][1]-intervals[0][0]
                success_tolerance = max(success_tolerance, 1./count)+1e-8
        if not close(old['cost'], new['cost']):
            reasons.append(side+'_cost_not_stable')
        for key, tolerance in (
            ('local_per_object_position_mean_m', settings.position_tolerance_m),
            ('local_per_object_orientation_mean_rad', settings.orientation_tolerance_rad),
            ('local_object_success_robust_fraction', success_tolerance),
            ('local_cws_tolerance_error', settings.cost_relative_tolerance)):
            if not close(old['metrics'][key], new['metrics'][key], tolerance):
                reasons.append(side+'_'+key+'_not_stable')
    old_gain = previous['control']['cost']-previous['candidate']['cost']
    new_gain = current['control']['cost']-current['candidate']['cost']
    if not close(old_gain, new_gain):
        reasons.append('candidate_advantage_not_stable')
    if (old_gain > 1e-4) != (new_gain > 1e-4):
        reasons.append('candidate_advantage_changed_sign')
    return reasons
