from pathlib import Path
import json
import numpy as np
import jax,jax.numpy as jnp
from flash_chord.training.flash_sac.eigen import correlated_scale_tril
b=Path(__file__).resolve().parents[1];basis=jnp.asarray(np.load(b/'priors/sharpa_bimanual_residual_basis.npz')['factor']);rows=[]
@jax.jit
def check(logstd,logfactor):
 def loss(a,f):
  L=correlated_scale_tril(jnp.exp(a),basis*jnp.exp(f))
  return jnp.sum(jnp.log(jnp.diagonal(L,axis1=-2,axis2=-1)))+jnp.sum(L**2)
 L=correlated_scale_tril(jnp.exp(logstd),basis*jnp.exp(logfactor))
 return L,jax.grad(loss,argnums=(0,1))(logstd,logfactor)
for es in [.07,.1,.3,1.]:
 for ds in [.01,.001,.0001,float(np.exp(-10)*np.sqrt(.5))]:
  std=np.full((2,56),ds,np.float32);std[1,::2]=np.exp(2)
  L,grad=check(jnp.log(std),jnp.log(jnp.float32(es)));L=np.asarray(L)
  assert np.isfinite(L).all() and all(np.isfinite(g).all() for g in grad)
  assert np.all(np.diagonal(L,axis1=-2,axis2=-1)>0)
  cov=std[:,:,None]**2*np.eye(56)+(np.asarray(basis)*es)@(np.asarray(basis)*es).T
  err=float(np.max(np.abs(L@np.swapaxes(L,-1,-2)-cov)));assert err<5e-5,err
  rows.append(dict(eigen_std=es,diagonal_std=ds,finite=True,covariance_max_error=err))
(b/'eigen_extremes_validation.json').write_text(json.dumps({'passed':True,'cases':rows},indent=2));print('PASS: 16 conditioning cases, heterogeneous std, positive diagonal and finite gradients')
