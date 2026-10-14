"""Local increments with cumulative bounds; unrelated frames remain unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation


def local_envelope(frames, fps, start, end, shoulder=.5):
    if not (0 <= start < end and shoulder > 0):
        raise ValueError('invalid local interval')
    t = np.arange(frames) / fps
    distance = np.maximum(start-t, t-end)
    w = .5 + .5*np.cos(np.pi*np.clip(distance/shoulder, 0, 1))
    w[[0, -1]] = 0
    return w


def feasible_increment(current, increment, lower, upper, speed, fps):
    """Largest line-search step satisfying value and finite-difference bounds.

    Contract only the proposed increment, never the existing correction.
    Each call handles one coordinate; zeros outside its support stay zero.
    """
    c, d = np.asarray(current, float), np.asarray(increment, float)
    if not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError('nonfinite correction')
    lo, hi = np.broadcast_to(lower, c.shape), np.broadcast_to(upper, c.shape)
    alpha = 1.
    for value, direction, low, high in (
        (c, d, lo, hi),
        (np.diff(c)*fps, np.diff(d)*fps, -speed, speed),
    ):
        low, high = np.broadcast_to(low, value.shape), np.broadcast_to(high, value.shape)
        if np.any(value < low-2e-5) or np.any(value > high+2e-5):
            raise ValueError('existing correction violates cumulative bounds')
        plus, minus = direction > 1e-12, direction < -1e-12
        if plus.any(): alpha = min(alpha, float(np.min((high[plus]-value[plus])/direction[plus])))
        if minus.any(): alpha = min(alpha, float(np.min((low[minus]-value[minus])/direction[minus])))
    alpha = float(np.clip(alpha, 0, 1))
    return c + alpha*d, alpha


def local_candidate(editor, action):
    a = np.asarray(action, float)
    if a.shape != (editor.joint_basis.shape[0],) or not np.isfinite(a).all():
        raise ValueError('one finite coefficient per spatial channel required')
    raw = np.clip(a, -1, 1) @ editor.joint_basis
    base, current = editor.baseline.joint_q, editor.binding.joint_q
    q = current.copy(); p = editor.projector; fps = editor.baseline.fps
    envelope = editor.basis[:, 0]
    scales = []
    for i, (name, qid) in enumerate(zip(p.layout.scalar_joints.names, p.qids)):
        translation = name.endswith(('wrist_x', 'wrist_y', 'wrist_z'))
        bound = float(editor.options.get('wrist_translation_offset_m', .03)) if translation else .52
        speed = float(editor.options.get('wrist_translation_correction_speed_m_s', .10)) if translation else 2.
        step = float(editor.options.get('local_translation_step_m', .02)) if translation else .20
        lo, hi = np.maximum(-bound, p.lower[i]-base[:, qid]), np.minimum(bound, p.upper[i]-base[:, qid])
        delta, alpha = feasible_increment(current[:, qid]-base[:, qid], envelope*raw[qid]*step, lo, hi, speed, fps)
        q[:, qid] = base[:, qid] + delta; scales.append(alpha)
    for start, _ in p.ball_joints:
        origin = Rotation.from_quat(base[:, start:start+4])
        correction = (Rotation.from_quat(current[:, start:start+4])*origin.inv()).as_rotvec()
        new = correction.copy()
        for axis in range(3):
            new[:, axis], alpha = feasible_increment(correction[:, axis], envelope*raw[start+axis]*.20,
                -.52/np.sqrt(3), .52/np.sqrt(3), 1.5/np.sqrt(3), fps)
            scales.append(alpha)
        q[:, start:start+4] = (Rotation.from_rotvec(new)*origin).as_quat()
    # Preserve exact data, including quaternion signs, outside the edited interval.
    q[envelope == 0] = current[envelope == 0]
    editor.last_projection_scales = scales
    return q
