"""Contact-constrained hand IK with explicit landmark correspondence.

The caller owns contact semantics and forward kinematics.  This module only
needs a floating-wrist channel descriptor, scalar finger limits and a callback
returning landmark positions in the same coordinate system as the targets.
It provides geometric fits, not collision or dynamic feasibility certificates.
"""
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import HandChannels


@dataclass(frozen=True)
class ContactTarget:
    """One intended FK landmark and its desired surface/pad position."""

    landmark: int
    position: tuple[float, float, float]
    weight: float = 1.

    def __post_init__(self):
        if not isinstance(self.landmark, (int, np.integer)) or self.landmark < 0:
            raise ValueError('contact landmark must be a nonnegative integer')
        if np.shape(self.position) != (3,) or not np.isfinite(self.position).all():
            raise ValueError('contact position must be a finite 3D point')
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('contact weight must be finite and positive')


def select_distal_contacts(points, active, groups: Mapping[Hashable, Sequence[int]],
                           landmark_ids: Mapping[Hashable, int], *, object_ids=None,
                           object_id=None, normals=None, normal_offset_m=0.) -> tuple[ContactTarget, ...]:
    """Pick the most distal active contact in each explicitly mapped digit.

    ``groups`` contains contact-slot indices ordered from proximal to distal.
    For example an adapter may map three human phalanges to each robot digit,
    and map that digit to its FK fingertip index with ``landmark_ids``.  Slots
    are never assigned to whichever FK point happens to be closest.  A proximal
    and distal label on one digit do not average two different surface points.

    Optional part/object filtering must use the same ID convention on both
    arguments.  An outward normal offset can account for a fingertip site's
    distance from its contact pad, when the adapter has trustworthy normals.
    """
    p, a = np.asarray(points, float), np.asarray(active)
    if p.ndim != 2 or p.shape[1] != 3 or a.shape != (len(p),):
        raise ValueError('contact points and activation must have shapes [slot,3] and [slot]')
    valid = np.isfinite(a) & (a > 0) & np.isfinite(p).all(axis=1)
    if object_id is not None:
        ids = np.asarray(object_ids)
        if ids.shape != (len(p),):
            raise ValueError('object filtering requires one object ID per contact slot')
        valid &= ids == object_id
    elif object_ids is not None:
        raise ValueError('object_ids requires an explicit object_id')
    if not np.isfinite(normal_offset_m):
        raise ValueError('normal offset must be finite')
    n = None if normals is None else np.asarray(normals, float)
    if normal_offset_m:
        if n is None or n.shape != p.shape:
            raise ValueError('a normal offset requires one 3D normal per slot')
        lengths = np.linalg.norm(n, axis=1)
        valid &= np.isfinite(n).all(axis=1) & (lengths > 1e-10)
    result = []
    seen = set()
    for digit, slots in groups.items():
        if digit not in landmark_ids:
            raise ValueError(f'missing FK landmark for contact group {digit!r}')
        slots = tuple(slots)
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(p) for i in slots):
            raise ValueError('contact group index is outside the contact arrays')
        landmark = landmark_ids[digit]
        if landmark in seen:
            raise ValueError('each contact group requires a distinct FK landmark')
        seen.add(landmark)
        selected = next((i for i in reversed(slots) if valid[i]), None)
        if selected is not None:
            point = p[selected].copy()
            if normal_offset_m:
                point += normal_offset_m * n[selected] / lengths[selected]
            result.append(ContactTarget(landmark, tuple(point)))
    return tuple(result)


@dataclass(frozen=True)
class ContactIKConfig:
    max_translation_m: float = .04
    max_rotation_rad: float = .35
    max_finger_delta_rad: float = .7
    contact_tolerance_m: float = .01
    contact_scale_m: float = .005
    translation_regularization_m: float = .04
    rotation_regularization_rad: float = .35
    finger_regularization_rad: float = .7
    regularization_weight: float = .1
    translation_difference_m: float = .0002
    rotation_difference_rad: float = .002
    finger_difference_rad: float = .002
    optimize_translation: bool = True
    optimize_rotation: bool = True
    max_evaluations: int = 80

    def __post_init__(self):
        for name, value in vars(self).items():
            if name in ('optimize_translation', 'optimize_rotation', 'max_evaluations'):
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
            raise ValueError('max_evaluations must be a positive integer')


@dataclass(frozen=True)
class ContactIKResult:
    """Fit diagnostics; ``feasible`` checks positional contact tolerance only.

    Adapters must check their additional normal, collision and dynamic
    contracts independently; convergence does not certify those constraints.
    """

    joint_q: np.ndarray
    initial_errors_m: np.ndarray
    errors_m: np.ndarray
    translation_delta_m: np.ndarray
    rotation_delta_rad: np.ndarray
    finger_delta_rad: np.ndarray
    converged: bool
    feasible: bool
    cost: float
    evaluations: int
    message: str


def _bounded_vector(value, radius):
    return value * min(1., radius / max(float(np.linalg.norm(value)), 1e-15))


def fit_hand_contacts(seed_q, hand: HandChannels, targets: Sequence[ContactTarget],
                      forward_kinematics: Callable[[np.ndarray], np.ndarray],
                      lower, upper, *, config=ContactIKConfig(),
                      extra_residuals: Callable[[np.ndarray], np.ndarray] | None = None) -> ContactIKResult:
    """Fit wrist/fingers to their intended contacts with bounded pose deltas.

    ``lower`` and ``upper`` contain one scalar bound per ``hand.fingers``.
    ``forward_kinematics(q)`` returns ``[landmark,3]`` world positions for a full
    joint-q vector.  Other-hand and unrelated channels remain exactly unchanged.
    Translation and world rotation deltas obey Euclidean trust-region limits;
    fingers obey both joint limits and per-joint delta limits.  Fixed joints and
    disabled wrist variables are excluded from the numerical optimization.

    The numerical Jacobian uses *absolute*, bound-aware central differences.
    Its default 0.2 mm / 2 mrad perturbations survive FK implementations that
    cast q to float32, including at zero-valued optimization variables.  SciPy's
    relative default steps can silently yield zero contact derivatives there.

    ``extra_residuals(q)`` may append adapter-supplied dimensionless normal,
    collision-clearance or other geometric residuals. It runs immediately after
    FK, so an adapter can reuse the updated body transforms. It must return a
    finite vector with a constant length. This core assigns no episode-specific
    collision roles or normal directions.
    """
    seed = np.asarray(seed_q, float)
    indices = (*hand.position, *hand.quaternion, *hand.fingers)
    if seed.ndim != 1 or not np.isfinite(seed).all() or max(indices) >= len(seed):
        raise ValueError('seed must be a finite full joint-q vector containing all hand channels')
    if np.linalg.norm(seed[list(hand.quaternion)]) < 1e-10:
        raise ValueError('seed wrist quaternion must be nonzero')
    targets = tuple(targets)
    if not targets or any(not isinstance(target, ContactTarget) for target in targets):
        raise ValueError('at least one ContactTarget is required')
    ids = np.array([target.landmark for target in targets], int)
    if len(set(ids)) != len(ids):
        raise ValueError('each intended landmark must have exactly one target')
    points = np.array([target.position for target in targets], float)
    weights = np.sqrt([target.weight for target in targets])[:, None]
    fingers = np.array(hand.fingers, int)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != fingers.shape or hi.shape != fingers.shape or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError('finger bounds must match the finger channels and contain no NaNs')
    if np.any(lo > hi):
        raise ValueError('lower finger bounds must not exceed upper bounds')
    finger_lo = np.maximum(lo - seed[fingers], -config.max_finger_delta_rad)
    finger_hi = np.minimum(hi - seed[fingers], config.max_finger_delta_rad)
    if np.any(finger_lo > finger_hi):
        raise ValueError('finger joint limits do not intersect the seed trust region')
    bound = np.r_[np.full(3, config.max_translation_m if config.optimize_translation else 0.),
                  np.full(3, config.max_rotation_rad if config.optimize_rotation else 0.)]
    lb, ub = np.r_[-bound, finger_lo], np.r_[bound, finger_hi]
    free = np.flatnonzero(ub > lb)
    initial = np.clip(np.zeros(len(lb)), lb, ub)
    initial[ub == lb] = lb[ub == lb]
    difference = np.r_[np.full(3, config.translation_difference_m),
                       np.full(3, config.rotation_difference_rad),
                       np.full(len(fingers), config.finger_difference_rad)]
    # Trust-region reflective least squares moves an exactly bound-valued
    # initial coordinate only about 1e-10 into the interval.  With float32 FK
    # this can make its first trial motion invisible and terminate immediately.
    # Start free coordinates at least one meaningful FK perturbation inside
    # their bounds; fixed coordinates remain exact.
    margin = np.minimum(difference[free], .25 * (ub[free] - lb[free]))
    initial[free] = np.clip(initial[free], lb[free] + margin, ub[free] - margin)
    prior_scale = np.r_[np.full(3, config.translation_regularization_m),
                        np.full(3, config.rotation_regularization_rad),
                        np.full(len(fingers), config.finger_regularization_rad)]
    wrist_rotation = Rotation.from_quat(seed[list(hand.quaternion)])
    calls = 0
    extra_count = None

    def decode(x):
        delta = initial.copy()
        delta[free] = x
        delta[:3] = _bounded_vector(delta[:3], config.max_translation_m)
        delta[3:6] = _bounded_vector(delta[3:6], config.max_rotation_rad)
        q = seed.copy()
        q[list(hand.position)] += delta[:3]
        q[list(hand.quaternion)] = (Rotation.from_rotvec(delta[3:6]) * wrist_rotation).as_quat()
        q[fingers] += delta[6:]
        return q, delta

    def positions(q):
        nonlocal calls
        calls += 1
        p = np.asarray(forward_kinematics(q), float)
        if p.ndim != 2 or p.shape[1] != 3 or max(ids) >= len(p) or not np.isfinite(p[ids]).all():
            raise ValueError('FK must return finite [landmark,3] positions containing every target')
        return p[ids]

    def residual(x):
        nonlocal extra_count
        q, delta = decode(x)
        contact = ((positions(q) - points) * weights / config.contact_scale_m).ravel()
        extra = np.empty(0) if extra_residuals is None else np.asarray(extra_residuals(q), float)
        if extra.ndim != 1 or not np.isfinite(extra).all():
            raise ValueError('extra residuals must be a finite one-dimensional vector')
        if extra_count is not None and extra.size != extra_count:
            raise ValueError('extra residual count must remain constant during an IK fit')
        extra_count = extra.size
        return np.r_[contact, np.sqrt(config.regularization_weight) * delta / prior_scale, extra]

    def jacobian(x):
        columns = []
        for j, variable in enumerate(free):
            step = difference[variable]
            positive = min(step, ub[variable] - x[j])
            negative = min(step, x[j] - lb[variable])
            xp, xm = x.copy(), x.copy()
            xp[j] += positive
            xm[j] -= negative
            columns.append((residual(xp) - residual(xm)) / (positive + negative))
        return np.stack(columns, axis=1)

    initial_errors = np.linalg.norm(positions(seed) - points, axis=1)
    if len(free):
        solution = least_squares(residual, initial[free], jac=jacobian, bounds=(lb[free], ub[free]),
                                 x_scale=prior_scale[free], max_nfev=config.max_evaluations,
                                 ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q, delta = decode(solution.x)
        converged, cost, message = bool(solution.success), float(solution.cost), str(solution.message)
    else:
        q, delta = decode(np.empty(0))
        r = residual(np.empty(0))
        converged, cost, message = True, float(.5 * r @ r), 'all optimization channels are fixed'
    errors = np.linalg.norm(positions(q) - points, axis=1)
    return ContactIKResult(q, initial_errors, errors, delta[:3].copy(), delta[3:6].copy(), delta[6:].copy(),
                           converged, bool(np.all(errors <= config.contact_tolerance_m)), cost, calls, message)


def transfer_hand_delta(command_q, achieved_q, fitted_q, hand: HandChannels, *, weight=1.):
    """Apply measured IK correction to a command without replacing its tracking bias.

    This transfers the world wrist translation/rotation and scalar finger
    deltas from ``achieved_q`` to ``fitted_q``.  A caller still projects temporal
    velocity, joint and command trust-region limits before physical evaluation.
    """
    command, achieved, fitted = (np.asarray(q, float) for q in (command_q, achieved_q, fitted_q))
    if command.ndim != 1 or achieved.shape != command.shape or fitted.shape != command.shape:
        raise ValueError('command, achieved and fitted joint-q vectors must have matching shapes')
    if not np.isfinite([command, achieved, fitted]).all() or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('finite joint-q vectors and a weight in [0,1] are required')
    result = command.copy()
    if weight == 0:
        return result
    p, r, f = list(hand.position), list(hand.quaternion), list(hand.fingers)
    result[p] += weight * (fitted[p] - achieved[p])
    result[f] += weight * (fitted[f] - achieved[f])
    rotation_delta = Rotation.from_quat(fitted[r]) * Rotation.from_quat(achieved[r]).inv()
    result[r] = (Rotation.from_rotvec(weight * rotation_delta.as_rotvec())
                 * Rotation.from_quat(command[r])).as_quat()
    return result
