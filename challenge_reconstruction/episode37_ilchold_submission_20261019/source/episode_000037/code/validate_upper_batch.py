"""Exercise batched upper proposal generation and replay insertion on CPU."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
from flash_chord.training.flash_sac.learner import Transition
editor=TTSAReferenceEditor.__new__(TTSAReferenceEditor)
editor.upper_worlds=2048;editor.upper=None;editor.training=SimpleNamespace(seed=43)
editor.basis=np.zeros((12,2));editor.joint_basis=np.zeros((4,8))
obs=jnp.ones((2048,10));action,context=editor.prepare_upper(obs)
assert action.shape==(2048,8)
assert np.unique(np.asarray(action),axis=0).shape[0]==2048
transition=Transition(action=action,objective_terms=jnp.linspace(-1,1,2048)[:,None],
 terminated=jnp.ones(2048,jnp.int32),truncated=jnp.zeros(2048,jnp.int32),
 episode_reference_progress=jnp.ones(2048,jnp.float32),termination_diagnostics=jnp.zeros((2048,0)),
 next_observation=obs,next_critic_context=context,post_reset_observation=obs,post_reset_critic_context=context)
editor.upper,_,loss=editor.executables.consume_transition(editor.upper,transition,
 jnp.ones(1),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
jax.block_until_ready(editor.upper)
assert int(editor.upper.replay.size)==2048
assert int(editor.upper.global_update_step)>0
assert np.isfinite(float(loss.actor_loss_mean_per_update))
print({'passed':True,'distinct_proposals':2048,'replay_size':int(editor.upper.replay.size),'upper_updates':int(editor.upper.global_update_step)})
