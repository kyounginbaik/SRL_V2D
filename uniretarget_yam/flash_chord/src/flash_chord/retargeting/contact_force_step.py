"""Contact-force allocation followed by bounded robot reference displacement.

This is a quasistatic proposal model, not a simulator/training gradient. Every
step needs fresh physical validation. Empty or unsuitable contacts require a
contact-arrangement branch rather than pretending that local IK can fix them.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from .contact_wrench import project_contact_wrench


@dataclass(frozen=True)
class ContactForceStep:
    delta: np.ndarray
    target_contact_forces: np.ndarray
    target_pad_displacement: np.ndarray
    predicted_pad_displacement: np.ndarray
    missing_wrench: np.ndarray
    requires_contact_change: bool
    projection_converged: bool
    message: str


def propose_contact_force_step(patches, measured_forces, desired_hand_wrench,
        point_jacobians, variable_scales, lower_delta, upper_delta, *,
        normal_force_limits, contact_stiffness_n_per_m, center=(0.,0.,0.),
        length_scale_m=.05, trust_radius=1., damping=.01,
        effort_matrix=None, effort_lower=None, effort_upper=None):
    """Redistribute physically available forces, then invert pad kinematics.

    Input patches contain ACTUATED hand/object contacts only. The desired hand
    wrench must already subtract legitimate non-hand forces. Do not subtract
    a table force when the fixed object task requires separating from it.
    point_jacobians[i] maps reference parameter changes to WORLD hand-pad
    displacement at contact i. Force normals point ON THE OBJECT. Moving a pad
    along (desired_force - measured_force) is the linear-compliance hypothesis.

    Stiffness is explicit in N/m and must be identified/validated by the caller;
    this function does not claim a universal contact stiffness. Trust scales
    give one physical edit unit per parameter (m, rad, or seconds). Bounds must
    contain zero. All contact groups require explicit normal-force budgets.
    Optional effort constraints map force-cone edge weights to HAND efforts.
    """
    patches=tuple(patches); force=np.asarray(measured_forces,float)
    jac=np.asarray(point_jacobians,float); scales=np.asarray(variable_scales,float)
    lo,hi=np.asarray(lower_delta,float),np.asarray(upper_delta,float)
    stiffness=np.asarray(contact_stiffness_n_per_m,float)
    if (scales.ndim!=1 or len(scales)<1 or jac.shape!=(len(patches),3,len(scales))
            or force.shape!=(len(patches),3) or lo.shape!=scales.shape or hi.shape!=lo.shape
            or not all(np.isfinite(v).all() for v in (force,jac,scales,lo,hi,stiffness))
            or np.any(scales<=0) or np.any(lo>0) or np.any(hi<0)
            or np.any(lo>hi) or not np.isfinite(trust_radius) or trust_radius<=0
            or not np.isfinite(damping) or damping<=0):
        raise ValueError('finite contact/Jacobian arrays, positive scales and zero-containing bounds required')
    if stiffness.ndim==0: stiffness=np.full(len(patches),float(stiffness))
    if stiffness.shape!=(len(patches),) or np.any(stiffness<=0):
        raise ValueError('explicit positive contact stiffness per patch required')
    if set(normal_force_limits)!={p.group for p in patches}:
        raise ValueError('every actuated contact group requires an explicit force budget')
    projection=project_contact_wrench(patches,desired_hand_wrench,center=center,
        normal_force_limits=normal_force_limits,length_scale_m=length_scale_m,
        effort_matrix=effort_matrix,effort_lower=effort_lower,effort_upper=effort_upper)
    if not projection.converged:
        raise ValueError('contact-force projection did not converge: '+projection.message)
    desired=np.asarray(desired_hand_wrench,float)
    units=np.array([1.,1.,1.,length_scale_m,length_scale_m,length_scale_m])
    unresolved=float(np.linalg.norm(projection.missing_wrench/units))
    demand=max(1e-8,float(np.linalg.norm(desired/units)))
    requires_change=bool(not patches or unresolved>.1*demand)
    displacement=(projection.contact_forces-force)/stiffness[:,None]
    scaled_jac=jac*scales[None,None,:]
    free=(hi>lo); delta=np.zeros(len(scales))
    if len(patches) and free.any() and np.linalg.norm(scaled_jac)>1e-15:
        # Normalize data energy so extra sampled patch points do not silently
        # change the strength of the reference regularization.
        a=scaled_jac.reshape(-1,len(scales))[:,free]
        scale=max(float(np.linalg.norm(a,2)),1e-12)
        a=np.vstack([a/scale,np.sqrt(damping)*np.eye(free.sum())])
        b=np.r_[displacement.reshape(-1)/scale,np.zeros(free.sum())]
        lower=np.maximum(lo/scales,-trust_radius)[free]
        upper=np.minimum(hi/scales,trust_radius)[free]
        result=lsq_linear(a,b,bounds=(lower,upper),tol=1e-11,max_iter=500)
        if not result.success: raise ValueError('bounded reference displacement failed')
        normalized=np.zeros(len(scales));normalized[free]=result.x
        normalized*=min(1.,trust_radius/max(np.linalg.norm(normalized),1e-15))
        delta=normalized*scales
    predicted=np.einsum('nkd,d->nk',jac,delta)
    return ContactForceStep(delta,projection.contact_forces,displacement,predicted,
        projection.missing_wrench,requires_change,projection.converged,
        'contact arrangement cannot supply requested wrench' if requires_change
        else 'quasistatic force-guided proposal; physical validation required')
