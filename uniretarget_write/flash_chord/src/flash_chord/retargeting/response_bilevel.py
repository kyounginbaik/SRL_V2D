"""Numerically checked lower responses and response-aware outer estimators.

The lower objective is explicitly a ridge adaptation surrogate, not the full
neural RL loss. See docs/response_bilevel.txt for the formulation and proof.
All numerical certificates are evaluated in float64; they are residual-based
error bounds in real arithmetic, not interval-arithmetic verification.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RidgeResponse:
    weights: np.ndarray
    gradient_norm: np.ndarray
    parameter_error_bound: np.ndarray
    objective_gap_bound: np.ndarray
    regularization: float


def ridge_system(features, targets, regularization):
    """G(W)=||XW-Y||_F^2/(2N)+lambda*||W||_F^2/2; optional batch axes."""
    x, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    if x.ndim < 2 or y.ndim != x.ndim or x.shape[:-1] != y.shape[:-1]:
        raise ValueError('matching [..., samples, features/actions] arrays required')
    if not x.shape[-2] or not x.shape[-1] or not y.shape[-1]:
        raise ValueError('empty ridge data')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite ridge data')
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('strictly positive regularization required')
    n = x.shape[-2]
    h = np.swapaxes(x, -1, -2) @ x / n + regularization*np.eye(x.shape[-1])
    b = np.swapaxes(x, -1, -2) @ y / n
    return h, b


def ridge_certificate(hessian, rhs, weights, regularization):
    """Caller must supply a ridge system with H >= regularization * I."""
    h, b, w = (np.asarray(a, np.float64) for a in (hessian, rhs, weights))
    if (h.ndim < 2 or h.shape[-1] != h.shape[-2] or b.shape != w.shape
            or h.shape[:-1] != b.shape[:-1]):
        raise ValueError('incompatible ridge system shapes')
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('strictly positive regularization required')
    if not all(np.isfinite(a).all() for a in (h,b,w)):
        raise ValueError('nonfinite lower system')
    residual = h @ w - b
    r = np.linalg.norm(residual, axis=(-2,-1))
    return RidgeResponse(w, r, r/regularization, r*r/(2*regularization), regularization)


def solve_ridge(features, targets, regularization=1., *, error_tolerance=1e-8):
    """Unique lower minimizer, refusing an unresolved numerical solve."""
    if not np.isfinite(error_tolerance) or error_tolerance <= 0:
        raise ValueError('positive finite error tolerance required')
    h, b = ridge_system(features, targets, regularization)
    w = np.linalg.solve(h, b)
    result = ridge_certificate(h,b,w,regularization)
    # A single iterative-refinement correction handles moderate conditioning.
    if np.any(result.parameter_error_bound > error_tolerance):
        w = w + np.linalg.solve(h,b-h@w)
        result = ridge_certificate(h,b,w,regularization)
    if np.any(result.parameter_error_bound > error_tolerance):
        raise ArithmeticError('lower response failed its residual error bound')
    return result


def ridge_response_jvp(hessian, weights, hessian_tangent, rhs_tangent):
    """Exact implicit derivative H^-1(dB-dH W), without forming H^-1."""
    h, w, dh, db = (np.asarray(a,np.float64) for a in
                    (hessian, weights, hessian_tangent, rhs_tangent))
    return np.linalg.solve(h, db-dh@w)


def gaussian_response_gradient(directions, costs_plus, costs_minus, sigma):
    """Two-sided Gaussian estimator of the *composed response* objective.

    Every plus/minus cost must be evaluated AFTER its own lower solve. For iid
    N(0,I) directions this estimates the gradient of Gaussian smoothing; using
    frozen-policy costs instead changes the objective and is not interchangeable.
    """
    u = np.asarray(directions,np.float64)
    plus, minus = np.asarray(costs_plus,np.float64), np.asarray(costs_minus,np.float64)
    if u.ndim != 2 or not u.shape[0] or plus.shape != (len(u),) or minus.shape != plus.shape:
        raise ValueError('directions [pairs, dimensions] and one cost per sign required')
    if not np.isfinite(sigma) or sigma <= 0 or not all(np.isfinite(a).all() for a in (u,plus,minus)):
        raise ValueError('finite costs/directions and positive sigma required')
    samples = ((plus-minus)/(2*sigma))[:,None]*u
    return samples.mean(axis=0)


def gaussian_oracle_bias_bound(value_error_bound, dimensions, sigma):
    """If each value error is <=eps, gradient bias is <=eps*sqrt(d)/sigma.

    A lower-parameter residual bound is NOT a value-error bound without a
    justified Lipschitz constant for the outer objective in lower parameters.
    """
    if (not np.isfinite(value_error_bound) or value_error_bound < 0
            or dimensions < 1 or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('invalid bias-bound arguments')
    return float(value_error_bound*np.sqrt(dimensions)/sigma)
