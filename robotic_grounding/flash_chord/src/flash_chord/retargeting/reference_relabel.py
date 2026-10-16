"""Relabel fixed-state replay observations under a local robot reference edit.

Reference FK derivatives use feasible finite differences of the actual bounded
editor and Newton FK (one-sided at active limits). The state distribution stays fixed; this is an explicit
local approximation, not differentiation through contact simulation. Object
commands, measured state, past actions, and desired wrench remain immutable.
"""
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation


def quat_multiply(a, b):
    """wxyz quaternion product, with arbitrary batch axes."""
    w = a[..., :1]*b[..., :1] - jnp.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    xyz = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + jnp.cross(a[..., 1:], b[..., 1:])
    return jnp.concatenate((w, xyz), axis=-1)


def quat_rotate(q, v):
    t = 2*jnp.cross(q[..., 1:], v)
    return v + q[..., :1]*t + jnp.cross(q[..., 1:], t)


def binding_for_q(editor, q):
    from flash_chord.embodiments.binding import build_robot_reference_binding
    target = editor.baseline.joint_target.copy()
    target[:, editor.projector.dids] = q[:, editor.projector.qids]
    for qi, di in editor.projector.ball_joints:
        target[:, di:di+3] = Rotation.from_quat(q[:, qi:qi+4]).as_rotvec()
    return build_robot_reference_binding(editor.projector.builder, editor.projector.layout,
                                         q, target, editor.baseline.fps)


def reference_values(binding, sides):
    poses = [np.concatenate((binding.hand(s).wrist_pos_w, binding.hand(s).wrist_quat_w), axis=-1)
             for s in sides]
    fingers = [binding.hand(s).finger_joint_pos for s in sides]
    return np.concatenate(poses+fingers, axis=-1)


class ReferenceRelabel:
    def __init__(self, values, jacobian, layout, config, bounds=None):
        self.values, self.jacobian = jnp.asarray(values), jnp.asarray(jacobian)
        if bounds is None:
            bounds = (-np.ones(56), np.ones(56))
        self.lower, self.upper = map(jnp.asarray, bounds)
        if (self.lower.shape != (56,) or self.upper.shape != (56,) or
                not np.isfinite(np.asarray(bounds)).all() or
                np.any(np.asarray(self.lower) > 0) or np.any(np.asarray(self.upper) < 0)):
            raise ValueError('finite feasible coefficient bounds containing zero required')
        self.layout, self.config = layout, config
        if layout.num_hands != 2 or layout.total_arms:
            raise ValueError('hypergradient relabeling supports floating bimanual hands only')
        for name in ('wrist_pos_start', 'wrist_quat_start', 'command_wrist_start', 'command_finger_start'):
            if getattr(layout, name) < 0:
                raise ValueError(f'required observation block disabled: {name}')
        for name in ('wrist_position', 'wrist_orientation', 'command_wrist', 'command_finger'):
            if getattr(config, name).scale == 0:
                raise ValueError(f'required observation scale is zero: {name}')
        if self.jacobian.shape != (*self.values.shape, 56):
            raise ValueError('reference Jacobian must be [frame, reference coordinate, 56]')

    def parameterize(self, raw, radius):
        """Smooth actor output inside the feasible box, avoiding blocked directions.

        A neutral network selects the feasible midpoint. At an active joint
        limit this is a small inward edit, not an impossible outward edit.
        The separately tested zero-edit control remains exactly zero.
        """
        lo, hi = jnp.maximum(self.lower, -radius), jnp.minimum(self.upper, radius)
        return (lo+hi)/2 + (hi-lo)/2 * (raw/radius)

    def reference(self, frame, theta):
        theta = jnp.clip(theta, self.lower, self.upper)
        return self.values[frame] + jnp.einsum('brd,d->br', self.jacobian[frame], theta)

    def __call__(self, observation, frame, theta):
        layout, config = self.layout, self.config
        reference = self.reference(frame, theta)
        delta = reference-self.values[frame]
        obs = observation
        for hand in range(2):
            pos = obs[:, layout.wrist_pos_start+3*hand:layout.wrist_pos_start+3*hand+3]/config.wrist_position.scale
            quat = obs[:, layout.wrist_quat_start+4*hand:layout.wrist_quat_start+4*hand+4]/config.wrist_orientation.scale
            quat = quat/jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
            inverse = quat*jnp.array([1., -1., -1., -1.])
            ref = reference[:, 7*hand:7*hand+7]
            ref_quat = ref[:, 3:]/jnp.maximum(jnp.linalg.norm(ref[:, 3:], axis=-1, keepdims=True), 1e-8)
            if config.make_wrist_quat_unique:
                ref_quat = jnp.where(ref_quat[:, :1] < 0, -ref_quat, ref_quat)
            relative = jnp.concatenate((quat_rotate(inverse, ref[:, :3]-pos),
                                        quat_multiply(inverse, ref_quat)), axis=-1)
            start = layout.command_wrist_start+7*hand
            obs = obs.at[:, start:start+7].set(relative*config.command_wrist.scale)
        start, count = layout.command_finger_start, layout.total_fingers
        return obs.at[:, start:start+count].add(delta[:, 14:]*config.command_finger.scale)

    @classmethod
    def from_editor(cls, editor, strategy, epsilon=.01):
        """Use only the edited frames for the expensive FK stencil."""
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('positive finite difference radius required')
        from flash_chord.embodiments.binding import build_robot_reference_binding
        values = reference_values(editor.binding, strategy.layout.sides)
        derivative = np.zeros((*values.shape, 56), np.float32)
        frames = np.flatnonzero(editor.basis[:, 0] > 0)
        # local_candidate limits each scalar/tangent coefficient independently.
        # The +/- unit probes give its exact feasible interval, including
        # cumulative displacement, joint limits and temporal speed bounds.
        coordinates = list(editor.projector.qids) + [
            start+axis for start, _ in editor.projector.ball_joints for axis in range(3)]
        expected = np.eye(editor.binding.joint_q.shape[1])[coordinates]
        if expected.shape != (56, editor.binding.joint_q.shape[1]) or not np.array_equal(editor.joint_basis, expected):
            raise ValueError('feasible relabeling requires the full canonical 56D joint basis')
        limits = []
        for sign in (1., -1.):
            editor.candidate_joint_q(np.full(56, sign))
            limits.append(np.asarray(editor.last_projection_scales, np.float32).copy())
        hi, lo = limits[0], -limits[1]
        for column in range(56):
            stencil = []
            for sign, available in ((1., hi[column]), (-1., -lo[column])):
                radius = min(epsilon, available)
                if radius < 1e-6:
                    continue
                theta = np.zeros(56); theta[column] = sign*radius
                q = editor.candidate_joint_q(theta)[frames]
                target = editor.baseline.joint_target[frames].copy()
                target[:, editor.projector.dids] = q[:, editor.projector.qids]
                for qi, di in editor.projector.ball_joints:
                    target[:, di:di+3] = Rotation.from_quat(q[:, qi:qi+4]).as_rotvec()
                binding = build_robot_reference_binding(editor.projector.builder, editor.projector.layout,
                                                        q, target, editor.baseline.fps)
                row = reference_values(binding, strategy.layout.sides)
                for hand in range(2):
                    ids = slice(7*hand+3, 7*hand+7)
                    sign_match = np.sum(row[:, ids]*values[frames, ids], axis=-1, keepdims=True)
                    row[:, ids] *= np.where(sign_match < 0, -1, 1)
                stencil.append((row-values[frames])/(sign*radius))
            if stencil:
                derivative[frames, :, column] = np.mean(stencil, axis=0)
        return cls(values, derivative, strategy.layout, strategy.config, bounds=(lo, hi))

    def validate_zero(self, observation, frame, tolerance=3e-4):
        reconstructed = self(observation, frame, jnp.zeros(56))
        error = float(jnp.max(abs(reconstructed-observation)))
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f'reference replay does not match current binding: {error}')
        return error
