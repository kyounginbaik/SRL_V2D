"""Sharpa human-derived finger exploration for the tanh-Gaussian SAC actor.

Basis lives in local unsquashed residual coordinates. Covariance is diagonal
plus learned low-rank finger terms, independent across hands; wrists unchanged.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import jax.numpy as jnp

@lru_cache(maxsize=8)
def load_basis(path, sha256):
    p=Path(path)
    if not sha256 or hashlib.sha256(p.read_bytes()).hexdigest()!=sha256:
        raise ValueError('Eigen basis hash mismatch or missing provenance')
    with np.load(p,allow_pickle=False) as f:basis=f['factor'].copy()
    if basis.shape!=(56,18) or not np.isfinite(basis).all():raise ValueError('Expected finite 56x18 basis')
    wrist=[0,1,2,3,4,5,28,29,30,31,32,33]
    if np.any(basis[wrist]):raise ValueError('Eigen noise may only affect finger channels')
    if np.any(basis[6:28,9:]) or np.any(basis[34:56,:9]):raise ValueError('Hand factors must be disjoint')
    for block in [basis[6:28,:9],basis[34:56,9:]]:
        np.testing.assert_allclose(np.sum(block**2),22.,rtol=1e-5)
    basis.setflags(write=False)
    return basis

def correlated_scale_tril(std, factor):
    """QR square root of diagonal-plus-low-rank covariance, per hand.

    Forming the covariance first loses the small positive diagonal in float32
    when the independent noise collapses. QR operates on the noise factors
    directly, preserving the distribution without a variance floor or jitter.
    """
    result=std[..., :,None]*jnp.eye(56,dtype=std.dtype)
    for start,rank_start in [(6,0),(34,9)]:
        f=factor[start:start+22,rank_start:rank_start+9]
        diagonal=std[...,start:start+22,None]*jnp.eye(22,dtype=std.dtype)
        f=jnp.broadcast_to(f,diagonal.shape[:-2]+f.shape)
        factors=jnp.concatenate((diagonal,f),axis=-1)
        _,r=jnp.linalg.qr(jnp.swapaxes(factors,-1,-2),mode='reduced')
        signs=jnp.where(jnp.diagonal(r,axis1=-2,axis2=-1)<0,-1.,1.)
        lower=jnp.swapaxes(r,-1,-2)*signs[...,None,:]
        result=result.at[...,start:start+22,start:start+22].set(lower)
    return result
