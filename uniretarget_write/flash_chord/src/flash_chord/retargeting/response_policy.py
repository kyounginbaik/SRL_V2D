"""Candidate-specific convex adaptation of a frozen FlashSAC mean head.

This is an explicit lower surrogate: ridge-fit a bounded, frozen-critic SAC
logit-improvement target on the candidate's own rollout observations. It does
not claim to globally optimize the neural SAC return. The backbone, critic,
temperature and exploration covariance are fixed within each bilevel stage.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze, FrozenDict
from flash_chord.training.flash_sac.network import Actor, Critic, policy_sample_from_noise
from flash_chord.retargeting.response_bilevel import solve_ridge, ridge_system, ridge_certificate


def actor_features(actor, params, batch_stats, observation):
    distribution, captured = actor.apply(
        {'params': params, 'batch_stats': batch_stats}, observation, training=False,
        capture_intermediates=lambda module, method: module.name == 'post_norm',
        mutable=['intermediates'])
    return distribution, captured['intermediates']['post_norm']['__call__'][0]


@dataclass
class HeadResponses:
    kernel: np.ndarray
    gain: np.ndarray
    bias: np.ndarray
    weights: np.ndarray
    certificate: object
    data_sha256: list[str]

    def params_for(self, original, index):
        params = unfreeze(original) if isinstance(original,FrozenDict) else jax.tree.map(lambda x:x,original)
        params['mean'] = {'unit_kernel':jnp.asarray(self.kernel[index]), 'gain':jnp.asarray(self.gain[index])}
        params['mean_bias'] = jnp.asarray(self.bias[index])
        return freeze(params) if isinstance(original,FrozenDict) else params


def fold_responses(original, response, features, targets, regularization, tolerance):
    """Preserve the existing actor architecture and check deployed head error."""
    old_kernel = np.asarray(original['mean']['unit_kernel'],np.float64)
    old_gain = np.asarray(original['mean']['gain'],np.float64)
    old_bias = np.asarray(original['mean_bias'],np.float64)
    effective = old_kernel*old_gain
    requested = effective[None]+response.weights[:,:-1]
    gain = np.linalg.norm(requested,axis=1).astype(np.float32)
    kernel = (requested/np.maximum(gain[:,None,:],1e-30)).astype(np.float32)
    zero = gain == 0
    kernel = np.where(zero[:,None,:],old_kernel[None],kernel).astype(np.float32)
    bias = (old_bias[None]+response.weights[:,-1]).astype(np.float32)
    deployed = np.concatenate((kernel.astype(np.float64)*gain[:,None,:]-effective[None],
                               (bias.astype(np.float64)-old_bias)[...,None,:]),axis=1)
    h,b=ridge_system(features,targets,regularization)
    certificate=ridge_certificate(h,b,deployed,regularization)
    if np.any(certificate.parameter_error_bound > tolerance):
        raise ArithmeticError('float32 policy head exceeds lower-response error tolerance')
    hashes=[hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for x,y in zip(features,targets)]
    return HeadResponses(kernel,gain,bias,deployed,certificate,hashes)


class ConvexHeadAdapter:
    def __init__(self, training, *, regularization=1., teacher_step=.1,
                 target_clip=.1, error_tolerance=1e-5, sample_stride=4):
        if (training.network.actor_head_mode != 'residual'
                or training.network.residual_mean_head != 'unit_gain'
                or training.network.action_transform != 'tanh'):
            raise ValueError('response adapter requires residual unit_gain tanh actor')
        if not all(np.isfinite(x) and x > 0 for x in
                   (regularization,teacher_step,target_clip,error_tolerance)) or sample_stride<1:
            raise ValueError('positive finite adaptation settings required')
        self.training=training; self.regularization=float(regularization)
        self.teacher_step=float(teacher_step);self.target_clip=float(target_clip)
        self.error_tolerance=float(error_tolerance);self.sample_stride=int(sample_stride)
        self.actor=Actor(action_dim=56,config=training.network,dtype=jnp.float32)
        lo,hi=training.reward_normalization.value_support
        critic=Critic(value_min=lo,value_max=hi,config=training.network,dtype=jnp.float32)

        def targets(params,stats,critic_params,critic_stats,temperature,obs,context,noise):
            distribution,features=actor_features(self.actor,params,stats,obs)
            def objective(mean):
                sample=policy_sample_from_noise(distribution.replace(mean=mean),noise,
                    action_transform='tanh',action_scale=training.network.action_scale)
                q=critic.apply({'params':critic_params,'batch_stats':critic_stats},
                    jnp.concatenate((obs,context),axis=-1),sample.action,training=False).q
                # Sum, not mean: each independent row receives its own gradient.
                return jnp.sum(temperature*sample.log_probability-jnp.minimum(q[0],q[1]))
            gradient=jax.grad(objective)(distribution.mean)
            return jnp.concatenate((features,jnp.ones((len(obs),1))),axis=1),gradient
        self.targets=jax.jit(targets)

        def policy(params,stats,obs,kernel,gain,bias):
            _,features=actor_features(self.actor,params,stats,obs)
            mean=jnp.einsum('bf,bfa->ba',features,kernel)*gain+bias
            return jnp.tanh(mean)*training.network.action_scale
        self.policy=jax.jit(policy)

    def collector(self, learner, seed):
        rows=[]
        def collect(frame, observation, context):
            if frame % self.sample_stride:
                return
            # Common random numbers across candidate worlds, fixed by frame.
            key=jax.random.fold_in(jax.random.PRNGKey(seed),frame)
            noise=jnp.broadcast_to(jax.random.normal(key,(1,56)),(len(observation),56))
            x,y=self.targets(learner.actor.params,learner.actor.batch_stats,
                learner.critic.params,learner.critic.batch_stats,
                jnp.exp(learner.temperature.log_temperature),observation,context,noise)
            x,y=np.asarray(x),np.asarray(y)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise FloatingPointError('nonfinite lower adaptation data')
            rows.append((x,y))
        return collect,rows

    def fit(self, learner, rows):
        if not rows:raise ValueError('no candidate-conditioned lower data collected')
        x=np.stack([r[0] for r in rows],axis=1).astype(np.float64)
        gradient=np.stack([r[1] for r in rows],axis=1).astype(np.float64)
        # One scalar step per candidate preserves the complete gradient
        # direction; coordinate clipping would change the proximal model.
        maximum=np.max(np.abs(gradient),axis=(1,2),keepdims=True)
        teacher_steps=np.minimum(self.teacher_step,self.target_clip/np.maximum(maximum,1e-30))
        y=-teacher_steps*gradient
        response=solve_ridge(x,y,self.regularization,error_tolerance=self.error_tolerance*.01)
        folded=fold_responses(learner.actor.params,response,x,y,self.regularization,self.error_tolerance)
        folded.teacher_steps=teacher_steps[:,0,0]
        return folded

    def policy_for(self, learner, responses):
        kernel,gain,bias=(jnp.asarray(x) for x in (responses.kernel,responses.gain,responses.bias))
        def act(observation,key):
            return self.policy(learner.actor.params,learner.actor.batch_stats,observation,kernel,gain,bias),key
        return act
