"""EigenDEXplore covariance and SAC density retain full finger control."""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.training.flash_sac.config import NetworkConfig
from flash_chord.training.flash_sac.eigen import correlated_scale_tril, load_basis
from flash_chord.training.flash_sac.network import Actor, policy_sample_from_noise


def basis_file(tmp_path):
    random = np.random.default_rng(5)
    block, _ = np.linalg.qr(random.normal(size=(22, 9)))
    block *= np.sqrt(np.linspace(1., .2, 9))[None, :]
    block *= np.sqrt(22. / np.sum(block**2))
    basis = np.zeros((56, 18), np.float32)
    basis[6:28, :9] = block
    basis[34:56, 9:] = block
    path = tmp_path/'basis.npz'
    np.savez(path, factor=basis)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), basis


def test_actor_preserves_joint_mean_and_variance_budget_with_correlated_density(tmp_path):
    path, checksum, basis = basis_file(tmp_path)
    config = NetworkConfig(actor_blocks=1, actor_hidden_dim=8, expansion=2,
        actor_head_mode='residual', residual_mean_head='zero_dense',
        eigen_basis_path=str(path), eigen_basis_sha256=checksum,
        initial_normalized_std=.1, eigen_variance_fraction=.5)
    actor = Actor(action_dim=56, config=config)
    observations = jnp.zeros((2, 5))
    variables = actor.init(jax.random.key(3), observations, training=False)
    distribution = actor.apply(variables, observations, training=False)
    lower = np.asarray(distribution.scale_tril)[0]
    covariance = lower @ lower.T
    expected = np.diag(np.asarray(distribution.std)[0]**2) + .005*(basis @ basis.T)
    np.testing.assert_allclose(covariance, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.trace(covariance[6:28, 6:28]), 22*.01, rtol=1e-5)
    np.testing.assert_allclose(covariance[:6, :6], .01*np.eye(6), atol=1e-8)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.max(np.abs(covariance[6:28, 6:28] - np.diag(np.diag(covariance[6:28, 6:28])))) > 1e-4
    # A mean direction outside the PCA span still reaches the finger target.
    direction = np.zeros(22)
    direction[0] = 1
    direction -= basis[6:28, :9] @ np.linalg.lstsq(basis[6:28, :9], direction, rcond=None)[0]
    bias = np.zeros(56, np.float32)
    bias[6:28] = direction*.1
    variables['params']['mean_bias'] = jnp.asarray(bias)
    shifted = actor.apply(variables, observations, training=False)
    np.testing.assert_allclose(np.asarray(shifted.mean-distribution.mean), np.broadcast_to(bias, (2,56)), atol=1e-7)
    noise = jax.random.normal(jax.random.key(9), shifted.mean.shape)
    sample = policy_sample_from_noise(shifted, noise, action_transform='tanh', action_scale=1.)
    residual = np.asarray(sample.raw_action-shifted.mean)[0]
    expected_log = -.5*(56*np.log(2*np.pi) + np.linalg.slogdet(covariance)[1]
                       + residual @ np.linalg.solve(covariance, residual))
    expected_log -= np.log(1-np.tanh(np.asarray(sample.raw_action)[0])**2).sum()
    np.testing.assert_allclose(float(sample.log_probability[0]), expected_log, atol=3e-5)


def test_small_independent_noise_keeps_factor_and_gradients_finite(tmp_path):
    _, _, basis = basis_file(tmp_path)
    factor = jnp.asarray(basis)*.1
    def loss(log_std):
        lower = correlated_scale_tril(jnp.exp(log_std), factor)
        return jnp.log(jnp.diag(lower)).sum()
    value, gradient = jax.value_and_grad(loss)(jnp.full((56,), np.log(1e-6)))
    assert np.isfinite(value) and np.isfinite(gradient).all()


def test_basis_provenance_and_finger_only_contract_are_enforced(tmp_path):
    path, checksum, _ = basis_file(tmp_path)
    assert load_basis(str(path), checksum).shape == (56, 18)
    with pytest.raises(ValueError, match='hash'):
        load_basis(str(path), 'invalid')
    basis = np.load(path)['factor']
    basis[0, 0] = 1
    np.savez(path, factor=basis)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='finger'):
        load_basis(str(path), checksum)
