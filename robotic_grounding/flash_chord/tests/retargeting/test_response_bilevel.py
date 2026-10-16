"""Independent numerical oracles for lower bounds and lower-response coupling."""
import numpy as np
import pytest
from flash_chord.retargeting.response_bilevel import (
    solve_ridge, ridge_system, ridge_certificate, ridge_response_jvp,
    gaussian_response_gradient, gaussian_oracle_bias_bound,
)


def test_rank_deficient_lower_has_unique_solution_and_valid_error_bounds():
    x = np.array([[1.,2.,1.],[2.,4.,2.],[3.,6.,3.]])
    y = np.array([[1.,-1.],[.5,2.],[3.,1.]])
    lam = .3
    response = solve_ridge(x,y,lam)
    # Independent SVD solution, not the normal-equations implementation.
    u,s,vt = np.linalg.svd(x,full_matrices=False)
    exact = vt.T @ ((s/(s*s+len(x)*lam))[:,None]*(u.T@y))
    np.testing.assert_allclose(response.weights,exact,atol=1e-12)
    h,b = ridge_system(x,y,lam)
    perturbed = exact + .07
    cert = ridge_certificate(h,b,perturbed,lam)
    error = np.linalg.norm(perturbed-exact)
    objective = lambda w: np.sum((x@w-y)**2)/(2*len(x)) + lam*np.sum(w*w)/2
    assert error <= cert.parameter_error_bound + 1e-13
    assert objective(perturbed)-objective(exact) <= cert.objective_gap_bound + 1e-13


def test_batched_solve_keeps_candidates_independent():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(3,20,5)); y=rng.normal(size=(3,20,2))
    out=solve_ridge(x,y,.2)
    for i in range(3):
        np.testing.assert_allclose(out.weights[i],solve_ridge(x[i],y[i],.2).weights)
    changed=y.copy(); changed[1]+=10
    other=solve_ridge(x,changed,.2)
    np.testing.assert_array_equal(out.weights[[0,2]],other.weights[[0,2]])
    assert not np.allclose(out.weights[1],other.weights[1])


def test_implicit_response_derivative_matches_finite_difference():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(30,5)); dx=rng.normal(size=x.shape)
    y=rng.normal(size=(30,2)); dy=rng.normal(size=y.shape)
    lam=.7; n=len(x)
    h,b=ridge_system(x,y,lam); w=solve_ridge(x,y,lam).weights
    implicit=ridge_response_jvp(h,w,(dx.T@x+x.T@dx)/n,(dx.T@y+x.T@dy)/n)
    eps=1e-5
    numerical=(solve_ridge(x+eps*dx,y+eps*dy,lam).weights-
               solve_ridge(x-eps*dx,y-eps*dy,lam).weights)/(2*eps)
    np.testing.assert_allclose(implicit,numerical,atol=1e-8,rtol=1e-7)


def test_outer_estimator_includes_response_missing_from_old_frozen_search():
    # G=.5*(w-2*theta)^2+.5*w^2 -> w*(theta)=theta.
    # F=(w-1)^2+.1*theta^2. At theta=0 true gradient=-2,
    # whereas holding w=0 gives zero and stalls the old method.
    rng=np.random.default_rng(3); u=rng.normal(size=(10000,1)); sigma=.05
    def adapted_cost(theta):
        w=solve_ridge(np.ones((1,1)),np.array([[2*theta]]),1.).weights.item()
        return (w-1)**2+.1*theta**2
    plus=np.array([adapted_cost(t) for t in sigma*u[:,0]])
    minus=np.array([adapted_cost(t) for t in -sigma*u[:,0]])
    g=gaussian_response_gradient(u,plus,minus,sigma)
    assert abs(g.item()+2) < .05
    frozen=gaussian_response_gradient(u,1+.1*(sigma*u[:,0])**2,
                                   1+.1*(-sigma*u[:,0])**2,sigma)
    assert frozen.item()==0
    assert adapted_cost(-.1*g.item()) < adapted_cost(0.)


def test_contracting_lower_iterations_approach_exact_response():
    rng=np.random.default_rng(4); x=rng.normal(size=(40,4));y=rng.normal(size=(40,2))
    lam=.5;h,b=ridge_system(x,y,lam);exact=solve_ridge(x,y,lam).weights
    lipschitz=np.linalg.eigvalsh(h).max(); w=np.zeros_like(exact)
    initial=np.linalg.norm(w-exact);rho=1-lam/lipschitz
    for k in range(1,31):
        w-= (h@w-b)/lipschitz
        assert np.linalg.norm(w-exact) <= initial*rho**k+1e-13


@pytest.mark.parametrize('lam',[0.,-1.,np.nan,np.inf])
def test_rejects_uncertifiable_regularization(lam):
    with pytest.raises(ValueError): solve_ridge(np.ones((3,2)),np.ones((3,1)),lam)


def test_bias_bound_and_invalid_cost_fail_closed():
    assert gaussian_oracle_bias_bound(.01,4,.1)==pytest.approx(.2)
    with pytest.raises(ValueError):
        gaussian_response_gradient(np.ones((2,3)),[1.,np.nan],[0.,0.],.1)
