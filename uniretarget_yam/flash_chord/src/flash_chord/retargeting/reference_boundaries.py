"""Reference boundary checks with explicit terminal finger-edit permission."""
import ast
import numpy as np


def validate_reference_boundaries(candidates, baseline, terminal_finger_q_ids=()):
    q = np.asarray(candidates)
    reference = np.asarray(baseline)
    if q.ndim != 3 or reference.shape != q.shape[1:] or q.shape[1] < 2:
        raise ValueError('matching candidate and baseline trajectories required')
    if not np.isfinite(q).all() or not np.isfinite(reference).all():
        raise ValueError('finite reference trajectories required')
    allowed = tuple(terminal_finger_q_ids)
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= q.shape[-1] for i in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError('unique valid terminal finger coordinate IDs required')
    np.testing.assert_allclose(q[:, 0], np.broadcast_to(reference[0], q[:, 0].shape), atol=1e-6)
    protected = sorted(set(range(q.shape[-1])) - set(allowed))
    np.testing.assert_allclose(q[:, -1][:, protected], np.broadcast_to(reference[-1, protected], (len(q), len(protected))), atol=1e-6)


def patch_rollout_text(source):
    marker = '# TERMINAL_FINGER_REFERENCE_V1'
    if marker in source:
        return source
    signature = 'def pack_candidate_references(env, q, target):'
    boundary = '''    for frame in [0,frames-1]:
        np.testing.assert_allclose(q[:,frame],np.broadcast_to(scene.robot_reference.joint_q[frame],(worlds,nq)),atol=1e-6)'''
    invocation = 'packed=pack_candidate_references(env,q,targets);install_candidate_references(env,packed)'
    if any(source.count(anchor) != 1 for anchor in [signature, boundary, invocation]):
        raise ValueError('unsupported rollout reference-boundary implementation')
    source = source.replace(signature, 'def pack_candidate_references(env, q, target, *, terminal_finger_q_ids=()):')
    source = source.replace(boundary, '''    # TERMINAL_FINGER_REFERENCE_V1
    from flash_chord.retargeting.reference_boundaries import validate_reference_boundaries
    validate_reference_boundaries(q,scene.robot_reference.joint_q,terminal_finger_q_ids)''')
    source = source.replace(invocation, "packed=pack_candidate_references(env,q,targets,terminal_finger_q_ids=getattr(editor,'terminal_finger_q_ids',()));install_candidate_references(env,packed)")
    ast.parse(source)
    return source
