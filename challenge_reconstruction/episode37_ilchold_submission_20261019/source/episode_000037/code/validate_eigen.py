from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import multivariate_normal
import jax,jax.numpy as jnp
from flash_chord.training.flash_sac.config import TrainingConfig,NetworkConfig,ReplayConfig,ExplorationConfig,RewardNormalizationConfig
from flash_chord.training.flash_sac.eigen import load_basis,correlated_scale_tril
from flash_chord.training.flash_sac.network import ActorDistribution,policy_sample_from_noise
from flash_chord.training.flash_sac.learner import create_learner,Transition
b=Path(__file__).resolve().parents[1];p=b/'priors/sharpa_bimanual_residual_basis.npz';sha=hashlib.sha256(p.read_bytes()).hexdigest();basis=load_basis(str(p),sha)
std=np.full(56,.1,np.float32);std[np.r_[6:28,34:56]]*=np.sqrt(.5);factor=basis*.1*np.sqrt(.5)
L=np.asarray(correlated_scale_tril(jnp.asarray(std),jnp.asarray(factor)));cov=np.diag(std**2)+factor@factor.T
np.testing.assert_allclose(L@L.T,cov,atol=1e-8)
noise=np.random.default_rng(43).normal(size=(128,56)).astype(np.float32)
d=ActorDistribution(mean=jnp.zeros_like(noise),log_std=jnp.log(jnp.asarray(std))[None,:]+jnp.zeros_like(noise),std=jnp.asarray(std)[None,:]+jnp.zeros_like(noise),scale_tril=jnp.asarray(L))
sample=policy_sample_from_noise(d,jnp.asarray(noise),action_transform='tanh',action_scale=1.)
expected=multivariate_normal.logpdf(np.asarray(sample.raw_action),cov=cov)-np.log1p(-np.asarray(sample.action,dtype=np.float64)**2).sum(-1)
np.testing.assert_allclose(np.asarray(sample.log_probability),expected,atol=2e-5,rtol=2e-6)
assert np.max(np.abs(cov[:6,:6]-.01*np.eye(6)))<1e-8
assert np.max(np.abs(cov[6:28,34:]))==0
cfg=TrainingConfig(seed=43,world_count=16,total_environment_steps=16000,updates_per_collection=2,mixed_precision=False,
 network=NetworkConfig(actor_head_mode='residual',residual_mean_head='zero_dense',actor_hidden_dim=32,critic_hidden_dim=64,eigen_basis_path=str(p),eigen_basis_sha256=sha),
 replay=ReplayConfig(capacity=256,minimum_size=4,batch_size=16,n_step=1),exploration=ExplorationConfig(warmup_action='policy',maximum_noise_repeat=1),reward_normalization=RewardNormalizationConfig(enabled=False,normalized_return_bound=10.))
obs=jnp.ones((16,10));ctx=jnp.ones((16,3));state,exe=create_learner(cfg,obs,ctx,action_dim=56,objective_term_count=1)
original=np.asarray(state.actor.params['eigen_log_std']).copy()
for _ in range(3):
 action,explore,key=exe.initial_action(state.actor.params,state.actor.batch_stats,state.replay.size,obs,state.exploration,state.key)
 state=state.replace(exploration=explore,key=key)
 t=Transition(action=action,objective_terms=jnp.linspace(-1,1,16)[:,None],terminated=jnp.ones(16,jnp.int32),truncated=jnp.zeros(16,jnp.int32),episode_reference_progress=jnp.ones(16),termination_diagnostics=jnp.zeros((16,0)),next_observation=jnp.array(obs,copy=True),next_critic_context=jnp.array(ctx,copy=True),post_reset_observation=jnp.array(obs,copy=True),post_reset_critic_context=jnp.array(ctx,copy=True))
 state,_,loss=exe.consume_transition(state,t,jnp.ones(1),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True));jax.block_until_ready(state)
 assert np.isfinite(float(loss.actor_loss_mean_per_update))
assert np.max(np.abs(np.asarray(state.actor.params['eigen_log_std'])-original))>0
report={'passed':True,'correlated_logpdf_max_error':float(np.max(np.abs(np.asarray(sample.log_probability)-expected))),'learner_updates':int(state.global_update_step),'eigen_parameter_max_change':float(np.max(np.abs(np.asarray(state.actor.params['eigen_log_std'])-original))),'wrists_unchanged':True,'hands_independent':True}
(b/'eigen_validation.json').write_text(json.dumps(report,indent=2));print(report)
