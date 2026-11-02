"""Native pyramidal contact-force feasibility using finite physical patch points."""
from dataclasses import dataclass
from typing import Hashable

import numpy as np
from scipy.optimize import linprog


# HiGHS supports feasibility tolerances down to 1e-10. Post-checks allow a
# small numerical margin, while requested probes must be larger again. Values
# apply after force/torque scaling, never directly to mixed N/Nm coordinates.
_LP_FEASIBILITY_TOLERANCE = 1e-10
_WRENCH_VALIDATION_TOLERANCE = 1e-9
_MINIMUM_RESOLVABLE_WRENCH = 10 * _WRENCH_VALIDATION_TOLERANCE


def _wrench_row_scales(target, length_scale_m):
    force_scale = max(1., float(np.abs(target[:3]).max(initial=0.)),
                      float(np.abs(target[3:]).max(initial=0.)) / length_scale_m)
    return force_scale * np.array([1., 1., 1., length_scale_m, length_scale_m, length_scale_m])


@dataclass(frozen=True)
class ContactPatch:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent1: tuple[float, float, float]
    tangent2: tuple[float, float, float]
    friction: tuple[float, float]
    group: Hashable

    def __post_init__(self):
        point = np.asarray(self.position, float)
        frame = np.asarray([self.normal, self.tangent1, self.tangent2], float)
        friction = np.asarray(self.friction, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('a finite contact position is required')
        if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-5)):
            raise ValueError('native contact normal and tangents must form an orthonormal frame')
        if friction.shape != (2,) or not np.isfinite(friction).all() or np.any(friction < 0):
            raise ValueError('two finite nonnegative sliding friction coefficients are required')
        hash(self.group)


@dataclass(frozen=True)
class WrenchFeasibility:
    feasible: bool
    normal_force_per_group: dict
    contact_forces: np.ndarray
    wrench_residual: np.ndarray
    minimum_total_normal_force: float | None
    message: str


def _wrench_columns(patches, origin):
    forces, moments, groups = [], [], []
    for patch in patches:
        n, t1, t2 = (np.asarray(v, float) for v in (patch.normal, patch.tangent1, patch.tangent2))
        arm = np.asarray(patch.position) - origin
        m1, m2 = patch.friction
        for force in (n + m1 * t1, n - m1 * t1, n + m2 * t2, n - m2 * t2):
            forces.append(force)
            moments.append(np.cross(arm, force))
            groups.append(patch.group)
    forces = np.asarray(forces).reshape(-1, 3)
    moments = np.asarray(moments).reshape(-1, 3)
    return forces, np.concatenate([forces.T, moments.T], axis=0), groups


def contact_wrench_support(patches, target_wrench, perturbation_scales, *, center=(0., 0., 0.),
                           normal_force_limits=None, length_scale_m=.05):
    """Diagnose missing wrench directions and probe local load tolerance.

    Supply the ACTUAL native contact cloud, not only its nominal grasp goals.
    Each positive perturbation scale is in N (first three) or Nm (last three).
    The twelve probes test target +/- one axis at a time; they are directional
    checks, not a certificate for every simultaneous perturbation or future
    contact configuration. Rank six alone is insufficient: unilateral contact,
    friction and force budgets still constrain the feasible wrench cone.

    SVD and LP checks scale torque by length_scale_m to avoid comparing N and
    Nm directly. Unsupported direction rows use those scaled coordinates.
    Probes below the reported numerical resolution raise ValueError instead
    of being silently treated as zero by the solver's feasibility tolerance.
    Every accepted probe also has componentwise residual below one percent of
    the requested increment (or the scaled solver tolerance, if tighter).
    """
    patches = tuple(patches)
    target, scales, origin = (np.asarray(x, float) for x in (target_wrench, perturbation_scales, center))
    if (target.shape != (6,) or scales.shape != (6,) or origin.shape != (3,)
            or not all(np.isfinite(x).all() for x in (target, scales, origin))
            or np.any(scales <= 0) or not np.isfinite(length_scale_m) or length_scale_m <= 0
            or any(not isinstance(patch, ContactPatch) for patch in patches)):
        raise ValueError('finite wrench/center, six positive probe scales, and positive length scale are required')
    _, matrix, _ = _wrench_columns(patches, origin)
    scaled = matrix / np.array([1., 1., 1., length_scale_m, length_scale_m, length_scale_m])[:, None]
    u, singular, _ = np.linalg.svd(scaled, full_matrices=True)
    rank = int((singular > singular.max(initial=0.) * 1e-9).sum())
    probes = np.stack([target + sign * np.diag(scales)[axis]
                       for axis in range(6) for sign in (-1., 1.)])
    # Use the largest requested wrench when checking probe resolution, so all
    # twelve directions are meaningful even when target/probe scales differ.
    row_scales = _wrench_row_scales(np.abs(probes).max(axis=0), length_scale_m)
    minimum_scales = _MINIMUM_RESOLVABLE_WRENCH * row_scales
    if np.any(scales < minimum_scales):
        raise ValueError('perturbation scales are below supported numerical precision; '
                         f'minimum force/torque increments are {minimum_scales.tolist()}')
    probe_tolerances = np.minimum(_WRENCH_VALIDATION_TOLERANCE * row_scales, .01 * scales)
    nominal = solve_contact_wrench(patches, target, center=origin, normal_force_limits=normal_force_limits,
                                  length_scale_m=length_scale_m)
    solutions = [solve_contact_wrench(patches, probe, center=origin,
                 normal_force_limits=normal_force_limits, length_scale_m=length_scale_m) for probe in probes]
    passed = np.array([solution.feasible and np.all(np.abs(solution.wrench_residual) <= probe_tolerances)
                       for solution in solutions])
    return dict(span_rank=rank, scaled_singular_values=singular,
        unsupported_scaled_wrench_directions=u[:, rank:].T, nominal=nominal,
        probe_wrenches=probes, probe_feasible=passed,
        minimum_perturbation_scales=minimum_scales,
        numerical_wrench_tolerances=_WRENCH_VALIDATION_TOLERANCE * row_scales,
        probe_wrench_tolerances=probe_tolerances,
        all_directional_probes_feasible=bool(nominal.feasible and passed.all()))


def solve_contact_wrench(patches, target_wrench, *, center=(0., 0., 0.), normal_force_limits=None,
                         length_scale_m=.05, normal_force_minima=None):
    """Find nonnegative native four-edge force weights balancing a 6D wrench.

    Positions, frames, center, target force [N] and target torque [Nm] must use
    the same coordinate frame. Normals point in the direction of force ON THE
    OBJECT. Use actual solver contact points and tangents; collapsing a finite
    patch to its centroid can falsely remove supported torques. This model is
    for condim=3 pyramidal contacts and includes no free torsional moments.

    Optional limits bound the sum of individual contact normal forces per
    group, so opposite forces cannot cancel the budget. Optional normal-force
    minima can allocate a self-balanced internal grip without changing the
    requested object wrench. Joint effort limits,
    motion feasibility and changing contacts require independent checks.

    Equality rows use force units after torque/length_scale_m conversion and
    normalization by max(1N, target force, target torque/length_scale_m).
    The scaled residual tolerance is 1e-9; force-budget tolerance uses the same
    force scale. A nonzero wrench entirely below the 1e-8 scaled resolution is
    rejected explicitly. Small components of a larger wrench remain subject
    to the stated residual tolerance, not an unsupported exactness claim.
    """
    patches = tuple(patches)
    target, origin = np.asarray(target_wrench, float), np.asarray(center, float)
    if (target.shape != (6,) or origin.shape != (3,) or not np.isfinite(target).all()
            or not np.isfinite(origin).all() or not np.isfinite(length_scale_m) or length_scale_m <= 0):
        raise ValueError('a finite 6D target wrench, 3D moment center and positive length scale are required')
    row_scales = _wrench_row_scales(target, length_scale_m)
    scaled_size = np.max(np.abs(target) / row_scales)
    if 0 < scaled_size < _MINIMUM_RESOLVABLE_WRENCH:
        raise ValueError('target wrench is below supported numerical precision')
    if any(not isinstance(patch, ContactPatch) for patch in patches):
        raise ValueError('each physical contact must be an explicit ContactPatch')
    if not patches:
        return WrenchFeasibility(False, {}, np.empty((0, 3)), -target, None, 'no physical contact patches')
    groups = tuple(dict.fromkeys(patch.group for patch in patches))
    limits = {} if normal_force_limits is None else dict(normal_force_limits)
    minima = {} if normal_force_minima is None else dict(normal_force_minima)
    if ((set(limits)|set(minima)) - set(groups)
            or any(not np.isfinite(v) or v < 0 for v in (*limits.values(),*minima.values()))
            or any(value>limits.get(g,np.inf) for g,value in minima.items())):
        raise ValueError('normal force bounds require known groups and finite ordered nonnegative values')
    force_columns, wrench_matrix, column_groups = _wrench_columns(patches, origin)
    group_rows = {group: np.array([name == group for name in column_groups], float) for group in groups}
    force_scale = row_scales[0]
    rows = np.array([group_rows[g] for g in limits]+[-group_rows[g] for g in minima]) / force_scale if (limits or minima) else None
    force_bounds = np.r_[list(limits.values()),[-v for v in minima.values()]] / force_scale if (limits or minima) else None
    solution = linprog(np.ones(len(force_columns)), A_ub=rows,b_ub=force_bounds,
        A_eq=wrench_matrix / row_scales[:, None], b_eq=target / row_scales,
        bounds=(0., None), method='highs', options={
            'primal_feasibility_tolerance': _LP_FEASIBILITY_TOLERANCE,
            'dual_feasibility_tolerance': _LP_FEASIBILITY_TOLERANCE})
    if not solution.success:
        return WrenchFeasibility(False, {}, np.zeros((len(patches), 3)), -target, None, str(solution.message))
    force_tolerance = _WRENCH_VALIDATION_TOLERANCE * force_scale
    weights = np.maximum(solution.x, 0.)
    forces = (weights[:, None] * np.asarray(force_columns)).reshape(len(patches), 4, 3).sum(1)
    residual = wrench_matrix @ weights - target
    group_forces = {group: float(row @ weights) for group, row in group_rows.items()}
    valid = bool(np.isfinite(weights).all() and np.min(solution.x) >= -force_tolerance
                 and np.max(np.abs(residual) / row_scales) <= _WRENCH_VALIDATION_TOLERANCE
                 and all(group_forces[group] <= limit + force_tolerance for group, limit in limits.items())
                 and all(group_forces[group] >= minimum - force_tolerance for group, minimum in minima.items()))
    message = str(solution.message) if valid else 'solver result violates scaled wrench or normal-force tolerance'
    return WrenchFeasibility(valid, group_forces, forces, residual, float(weights.sum()), message)


@dataclass(frozen=True)
class WrenchProjection:
    """Closest static wrench, not a gradient through the simulator."""
    converged: bool
    contact_forces: np.ndarray
    edge_weights: np.ndarray
    achieved_wrench: np.ndarray
    missing_wrench: np.ndarray
    scaled_loss: float
    normal_force_per_group: dict
    joint_effort: np.ndarray
    message: str


def project_contact_wrench(patches, target_wrench, *, center=(0., 0., 0.),
                           normal_force_limits=None, length_scale_m=.05,
                           effort_matrix=None, effort_lower=None, effort_upper=None):
    """Project a desired object wrench into the available contact force set.

    Minimize half the squared scaled wrench error under native four-edge
    friction cones, group normal-force limits, and optional joint efforts.
    ``effort_matrix`` maps the 4*N edge weights to HAND joint effort. The caller
    must supply signed contact Jacobians, including the reaction on the hand,
    and bounds after accounting for gravity/inertia/other loads. No absent
    effort constraint is inferred. A feasible *static* projection is not proof
    that a reference trajectory or current controller can realize its forces.

    Missing wrench is desired minus achieved (N, Nm). In particular a downward
    pressing contact cannot be credited with upward lift. With no contacts the
    zero projection converges but the full target remains missing. Torque rows
    use length_scale_m to avoid mixing N with Nm in the loss.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize
    patches = tuple(patches)
    target, origin = np.asarray(target_wrench, float), np.asarray(center, float)
    if (target.shape != (6,) or origin.shape != (3,) or not np.isfinite(target).all()
            or not np.isfinite(origin).all() or not np.isfinite(length_scale_m) or length_scale_m <= 0
            or any(not isinstance(p, ContactPatch) for p in patches)):
        raise ValueError('finite wrench/center, physical patches and positive length scale required')
    groups = tuple(dict.fromkeys(p.group for p in patches))
    limits = {} if normal_force_limits is None else dict(normal_force_limits)
    if set(limits) - set(groups) or any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('normal force limits require known groups and finite nonnegative values')
    force_columns, matrix, column_groups = _wrench_columns(patches, origin)
    n = matrix.shape[1]
    group_rows = {g: np.array([name == g for name in column_groups], float) for g in groups}
    supplied = (effort_matrix is not None, effort_lower is not None, effort_upper is not None)
    if any(supplied) and not all(supplied):
        raise ValueError('effort matrix and both bounds must be supplied together')
    effort = np.empty((0, n)) if not any(supplied) else np.asarray(effort_matrix, float)
    lower = np.empty(0) if not any(supplied) else np.asarray(effort_lower, float)
    upper = np.empty(0) if not any(supplied) else np.asarray(effort_upper, float)
    if (effort.ndim != 2 or effort.shape[1] != n or lower.shape != (effort.shape[0],)
            or upper.shape != lower.shape or not all(np.isfinite(v).all() for v in (effort, lower, upper))
            or np.any(lower > upper)):
        raise ValueError('finite effort map and ordered bounds with one column per force edge required')
    scales = _wrench_row_scales(target, length_scale_m)
    a, b = matrix / scales[:, None], target / scales
    if n == 0:
        valid = bool(np.all(lower <= 0) and np.all(upper >= 0))
        return WrenchProjection(valid, np.empty((0, 3)), np.empty(0), np.zeros(6), target.copy(),
                                float(.5 * b @ b), {}, np.zeros(len(lower)), 'no contact patches')
    rows = np.array([group_rows[g] for g in limits]).reshape(-1, n)
    constraint_matrix = np.concatenate([rows, effort], axis=0)
    constraint_lower = np.concatenate([np.full(len(rows), -np.inf), lower])
    constraint_upper = np.concatenate([np.array(list(limits.values()), float), upper])
    constraints = [] if not len(constraint_matrix) else [LinearConstraint(
        constraint_matrix, constraint_lower, constraint_upper)]
    # A tiny strictly convex tie breaker keeps redundant edge coefficients
    # bounded; it is reported separately from the physical mismatch loss.
    regularizer = 1e-12 / max(1., scales[0] ** 2)
    def objective(x):
        residual = a @ x - b
        return .5 * (residual @ residual + regularizer * (x @ x))
    def gradient(x):
        return a.T @ (a @ x - b) + regularizer * x
    result = minimize(objective, np.zeros(n), jac=gradient, method='SLSQP',
        bounds=Bounds(np.zeros(n), np.full(n, np.inf)), constraints=constraints,
        options=dict(ftol=1e-13, maxiter=2000))
    weights = np.maximum(result.x, 0.)
    constrained = constraint_matrix @ weights
    tolerance = 1e-7
    valid = bool(result.success and np.isfinite(weights).all()
        and np.all(constrained >= constraint_lower - tolerance)
        and np.all(constrained <= constraint_upper + tolerance))
    achieved = matrix @ weights
    missing = target - achieved
    return WrenchProjection(valid,
        (weights[:, None] * force_columns).reshape(len(patches), 4, 3).sum(1), weights,
        achieved, missing, float(.5 * np.sum((missing / scales) ** 2)),
        {g: float(row @ weights) for g, row in group_rows.items()}, effort @ weights,
        str(result.message) if valid else 'projection failed: ' + str(result.message))
