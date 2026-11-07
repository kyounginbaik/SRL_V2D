"""Fixed episode-state checks and historical successful-prefix protection."""
import hashlib
import numpy as np
from flash_chord.retargeting.failure_windows import failure_timing


def assert_initial_state(env, expected_sha256=None):
    """Check all robot/object generalized positions and zero velocities."""
    worlds=env.world_count
    q=env.state_0.joint_q.numpy().reshape(worlds,-1)
    qd=env.state_0.joint_qd.numpy().reshape(worlds,-1)
    expected=env.reset_table.ref_joint_q.numpy()[0].copy()
    offsets=env.reset_table.object_root_position_offsets_w.numpy().reshape(worlds,-1,3)
    np.testing.assert_array_equal(offsets,np.broadcast_to(offsets[0],offsets.shape))
    for root_id,offset in zip(env.reset_table.object_root_q_ids.numpy(),offsets[0]):
        expected[root_id:root_id+3]+=offset
    np.testing.assert_allclose(q,np.broadcast_to(expected,q.shape),rtol=0,atol=1e-6)
    np.testing.assert_array_equal(qd,np.zeros_like(qd))
    np.testing.assert_array_equal(env.timestep.numpy(),np.zeros(worlds,dtype=int))
    digest=hashlib.sha256(np.asarray(expected,dtype='<f4').tobytes()+
                          np.zeros(qd.shape[1],dtype='<f4').tobytes()).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('episode initial state changed between training/evaluation/refinement')
    return dict(initial_state_sha256=digest,worlds=worlds,frame=0,
                joint_position_max_error=float(np.max(abs(q-expected))),
                joint_velocity_max_error=float(np.max(abs(qd))))


def success_prefix_frames(p,a,fps,**timing_options):
    timing=failure_timing(p,a,fps,**timing_options)
    return timing['failure_frame'] if timing['failed'] else len(p)


def historical_prefix_check(position,orientation,incumbent_position,incumbent_orientation,
                            fps,**timing_options):
    p,a=np.asarray(position),np.asarray(orientation)
    current=success_prefix_frames(p,a,fps,**timing_options)
    if incumbent_position is None:
        return dict(saved_prefix_frames=0,current_success_prefix_frames=current,
                    saved_prefix_preserved=True,saved_prefix_rejection_reasons=[])
    bp,ba=np.asarray(incumbent_position),np.asarray(incumbent_orientation)
    if p.shape!=bp.shape or a.shape!=ba.shape:
        raise ValueError('saved demonstration has a different object roster or horizon')
    n=success_prefix_frames(bp,ba,fps,**timing_options)
    reasons=[]
    if current<n:
        reasons.append('earlier_failure_than_saved_demo')
    if n:
        for key,after,before,tolerance in (
                ('saved_prefix_position_mean_m',p[:n].mean(0),bp[:n].mean(0),.002),
                ('saved_prefix_orientation_mean_rad',a[:n].mean(0),ba[:n].mean(0),.01),
                ('saved_prefix_position_p95_m',np.percentile(p[:n],95,axis=0),np.percentile(bp[:n],95,axis=0),.005),
                ('saved_prefix_orientation_p95_rad',np.percentile(a[:n],95,axis=0),np.percentile(ba[:n],95,axis=0),.03)):
            if np.any(after>before+tolerance):reasons.append(key)
    return dict(saved_prefix_frames=n,current_success_prefix_frames=current,
                saved_prefix_preserved=not reasons,saved_prefix_rejection_reasons=reasons)


def better_demo(check,full_cost,incumbent_cost):
    """Prefer longer verified prefix, then lower fixed full-horizon objective."""
    if not check['saved_prefix_preserved'] or not np.isfinite(full_cost):return False
    return (incumbent_cost is None or check['current_success_prefix_frames']>check['saved_prefix_frames']
            or (check['current_success_prefix_frames']==check['saved_prefix_frames']
                and full_cost<incumbent_cost-1e-4))
