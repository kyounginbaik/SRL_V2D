import numpy as np
import jax
import jax.numpy as jnp
from flash_chord.training.flash_sac.config import TrainingConfig,NetworkConfig
from flash_chord.training.flash_sac.network import Actor,project_unit_parameters
from flash_chord.retargeting.response_bilevel import solve_ridge
from flash_chord.retargeting.response_policy import actor_features,fold_responses,ConvexHeadAdapter


def test_adapted_actor_matches_solved_head_and_preserves_other_parameters():
    config=NetworkConfig(actor_head_mode='residual',actor_hidden_dim=16,critic_hidden_dim=16)
    actor=Actor(action_dim=56,config=config)
    obs=jax.random.normal(jax.random.PRNGKey(1),(20,12))
    variables=actor.init(jax.random.PRNGKey(2),obs,training=False)
    params=project_unit_parameters(variables['params'])
    dist,h=actor_features(actor,params,variables['batch_stats'],obs)
    h=np.asarray(h);x=np.concatenate((h,np.ones((20,1))),axis=1)[None]
    y=np.random.default_rng(1).normal(0,.03,(1,20,56))
    lower=solve_ridge(x,y,1.)
    response=fold_responses(params,lower,x,y,1.,1e-5)
    updated=response.params_for(params,0)
    actual=actor.apply({'params':updated,'batch_stats':variables['batch_stats']},obs,training=False)
    expected=np.asarray(dist.mean)+x[0]@response.weights[0]
    np.testing.assert_allclose(actual.mean,expected,atol=1e-6)
    assert float(response.certificate.parameter_error_bound.max())<1e-5
    for name in params:
        if name not in ['mean','mean_bias']:
            for before,after in zip(jax.tree.leaves(params[name]),jax.tree.leaves(updated[name])):
                np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(actual.std,dist.std)


def test_batched_response_policy_matches_individual_folded_actors():
    config=TrainingConfig(world_count=3,total_environment_steps=3000000,network=NetworkConfig(actor_head_mode='residual',
        actor_hidden_dim=16,critic_hidden_dim=16))
    adapter=ConvexHeadAdapter(config)
    obs=jax.random.normal(jax.random.PRNGKey(3),(3,12))
    variables=adapter.actor.init(jax.random.PRNGKey(4),obs,training=False)
    params=project_unit_parameters(variables['params'])
    x=np.random.default_rng(3).normal(size=(3,25,17));y=np.ones((3,25,56))*.02
    responses=fold_responses(params,solve_ridge(x,y,1.),x,y,1.,1e-5)
    actual=adapter.policy(params,variables['batch_stats'],obs,
                          jnp.asarray(responses.kernel),jnp.asarray(responses.gain),jnp.asarray(responses.bias))
    for i in range(3):
        single=adapter.actor.apply({'params':responses.params_for(params,i),'batch_stats':variables['batch_stats']},
                                   obs[i:i+1],training=False)
        np.testing.assert_allclose(actual[i],jnp.tanh(single.mean[0]),atol=1e-6)


def test_real_sac_target_collector_produces_certified_candidate_responses():
    from types import SimpleNamespace as NS
    from flash_chord.training.flash_sac.network import Critic
    config=TrainingConfig(world_count=3,total_environment_steps=3000000,
        network=NetworkConfig(actor_head_mode='residual',actor_hidden_dim=16,critic_hidden_dim=16))
    adapter=ConvexHeadAdapter(config)
    obs=jax.random.normal(jax.random.PRNGKey(5),(3,12));context=jnp.zeros((3,3))
    actor_vars=adapter.actor.init(jax.random.PRNGKey(6),obs,training=False)
    lo,hi=config.reward_normalization.value_support
    critic=Critic(lo,hi,config.network)
    critic_vars=critic.init(jax.random.PRNGKey(7),jnp.concatenate((obs,context),axis=1),
                            jnp.zeros((3,56)),training=False)
    learner=NS(actor=NS(params=project_unit_parameters(actor_vars['params']),batch_stats=actor_vars['batch_stats']),
        critic=NS(params=project_unit_parameters(critic_vars['params']),batch_stats=critic_vars['batch_stats']),
        temperature=NS(log_temperature=jnp.array(-2.)))
    collect,rows=adapter.collector(learner,123)
    collect(0,obs,context);collect(4,obs*.8,context)
    responses=adapter.fit(learner,rows)
    assert responses.weights.shape==(3,17,56)
    assert np.linalg.norm(responses.weights)>0
    assert np.max(responses.certificate.parameter_error_bound)<adapter.error_tolerance
    assert not np.allclose(responses.weights[0],responses.weights[1])
