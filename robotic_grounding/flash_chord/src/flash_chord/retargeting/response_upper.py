"""FlashSAC proposals for the explicit candidate-response bilevel objective.

Each candidate is a terminal contextual-bandit transition: its reward is the
adapted no-edit cost minus its cost after solving its own lower response. No
critic bootstrap crosses a reference or lower-policy change. Replay is refreshed
at each search stage; network/optimizer parameters persist across stages.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.config import (
    TrainingConfig, NetworkConfig, OptimizationConfig, ReplayConfig,
    ExplorationConfig, RewardNormalizationConfig,
)
from flash_chord.training.flash_sac.learner import Transition, create_learner
from flash_chord.training.flash_sac.replay import clear_replay


class ResponseUpperFlashSAC:
    def __init__(self, seed, worlds, *, updates=32, initial_std=.5, action_dim=56):
        if worlds < 4 or updates < 2 or action_dim < 1:
            raise ValueError('Upper FlashSAC requires >=4 candidates and >=2 updates')
        self.worlds, self.action_dim = worlds, action_dim
        self.config = TrainingConfig(seed=seed, world_count=worlds,
            total_environment_steps=((1_000_000+worlds-1)//worlds)*worlds,
            updates_per_collection=updates, mixed_precision=False,
            network=NetworkConfig(actor_blocks=2, actor_hidden_dim=64,
                critic_blocks=2, critic_hidden_dim=128, actor_head_mode='residual',
                residual_mean_head='zero_dense', initial_normalized_std=initial_std,
                action_transform='identity'),
            optimization=OptimizationConfig(learning_rate_initial=1e-4,
                learning_rate_peak=1e-4, learning_rate_end=1e-4,
                learning_rate_schedule_updates=100_000),
            replay=ReplayConfig(capacity=max(1024, worlds*2), minimum_size=4,
                batch_size=min(256, max(16, worlds)), n_step=1),
            exploration=ExplorationConfig(warmup_action='policy', maximum_noise_repeat=1),
            reward_normalization=RewardNormalizationConfig(enabled=False, normalized_return_bound=10.))
        self.state = None
        self.pending = None

    def propose(self, observations):
        if self.pending is not None:
            raise RuntimeError('Score the previous proposal batch before proposing again')
        obs = jnp.asarray(observations, jnp.float32)
        if obs.ndim != 2 or obs.shape[0] != self.worlds or not np.isfinite(obs).all():
            raise ValueError('One finite contextual observation per candidate required')
        context = jnp.tile(jnp.array([[0., 0., 1.]], jnp.float32), (self.worlds, 1))
        if self.state is None:
            self.state, self.executables = create_learner(
                self.config, obs, context, action_dim=self.action_dim, objective_term_count=1)
        else:
            self.state = self.state.replace(replay=clear_replay(self.state.replay),
                observation=jnp.array(obs, copy=True), critic_context=jnp.array(context, copy=True))
        action, exploration, key = self.executables.initial_action(
            self.state.actor.params, self.state.actor.batch_stats, self.state.replay.size,
            obs, self.state.exploration, self.state.key)
        self.state = self.state.replace(exploration=exploration, key=key)
        action = action.at[0].set(0.)  # Same adapted no-edit control as Gaussian search.
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite upper policy proposals')
        self.pending = (obs, context, action)
        return np.array(action)

    def learn(self, adapted_costs):
        if self.pending is None:
            raise RuntimeError('No pending proposal batch')
        costs = np.asarray(adapted_costs, float)
        if costs.shape != (self.worlds,) or not np.isfinite(costs).all():
            raise ValueError('Finite solved-response cost required for every candidate')
        obs, context, action = self.pending
        rewards = np.clip(costs[0]-costs, -5., 5.).astype(np.float32)
        before_actor = jax.tree.map(lambda x: np.array(x, copy=True), self.state.actor.params)
        before_critic = jax.tree.map(lambda x: np.array(x, copy=True), self.state.critic.params)
        before_updates = int(self.state.global_update_step)
        transition = Transition(action=action, objective_terms=jnp.asarray(rewards[:, None]),
            terminated=jnp.ones(self.worlds, jnp.int32), truncated=jnp.zeros(self.worlds, jnp.int32),
            episode_reference_progress=jnp.ones(self.worlds, jnp.float32),
            termination_diagnostics=jnp.zeros((self.worlds, 0), jnp.float32),
            next_observation=obs, next_critic_context=context,
            post_reset_observation=obs, post_reset_critic_context=context)
        self.state, _, losses = self.executables.consume_transition(self.state, transition,
            jnp.ones(1, jnp.float32), jnp.float32(1), jnp.float32(0), jnp.bool_(False), jnp.bool_(True))
        jax.block_until_ready(self.state)
        self.pending = None
        def change(current, previous):
            return float(max(jax.tree.leaves(jax.tree.map(
                lambda a,b: np.max(np.abs(np.asarray(a)-b)), current, previous))))
        return dict(upper_optimizer_updates=int(self.state.global_update_step),
            upper_optimizer_updates_this_round=int(self.state.global_update_step)-before_updates,
            upper_actor_parameter_max_change=change(self.state.actor.params, before_actor),
            upper_critic_parameter_max_change=change(self.state.critic.params, before_critic),
            upper_actor_loss=float(losses.actor_loss_mean_per_update),
            upper_critic_loss=float(losses.critic_loss_mean_per_update),
            upper_actor_entropy=float(losses.actor_entropy_mean_per_update),
            upper_actor_updates_this_round=float(losses.actor_update_count),
            upper_actor_skipped_fraction=float(losses.actor_skipped_fraction),
            upper_critic_skipped_fraction=float(losses.critic_skipped_fraction),
            upper_reward_mean=float(rewards.mean()), upper_reward_max=float(rewards.max()),
            upper_replay_size=int(self.state.replay.size), upper_replay_refreshed=True,
            upper_reward='clip(adapted_no_edit_cost - candidate_solved_response_cost, -5, 5)')

    def configuration(self):
        return asdict(self.config)
