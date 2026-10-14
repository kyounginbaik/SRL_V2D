from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
from flash_chord.retargeting.failure_windows import failure_timing, propose_windows, trajectory_metrics, prefix_gate


def test_failure_detects_either_object_and_ignores_single_frame_spike():
    p = np.zeros((200, 2)); a = p.copy()
    p[5, 0] = .2
    p[32:40, 1] = .03
    p[40:, 1] = .06
    timing = failure_timing(p, a, 20)
    assert timing['failure_seconds'] == 2.
    assert timing['onset_seconds'] == 1.6
    assert timing['per_object_failure_seconds'] == [None, 2.]
    assert timing['detected_seconds'] == 2.1
    a[20:, 0] = .4
    assert failure_timing(p, a, 20)['failure_seconds'] == 1.


@pytest.mark.parametrize('failure_frame', [0, 1, 5, 20, 40, 160])
def test_early_failure_windows_fit_history_and_are_distinct(failure_frame):
    p = np.zeros((200, 2)); p[failure_frame:, 1] = .1
    timing = failure_timing(p, p*0, 20)
    windows = propose_windows(timing, 201, 20, np.random.default_rng(7), {})
    assert 1 <= len(windows) <= 4
    assert len({w['start_seconds'] for w in windows}) == len(windows)
    for w in windows:
        assert 0 <= w['start_seconds'] < w['end_seconds'] <= max(failure_frame/20, .05)+1e-8
        assert w['end_seconds']-w['start_seconds'] <= 1.00001
    if failure_frame < 20:
        assert len(windows) == 1


def test_sampling_reproducible_but_resamples_history_across_rounds():
    p = np.zeros((200, 2)); p[160:] = .1
    t = failure_timing(p, p*0, 20)
    def rounds():
        rng, visits = np.random.default_rng(5), {}
        return [propose_windows(t, 201, 20, rng, visits) for _ in range(3)]
    a, b = rounds(), rounds()
    assert a == b
    assert a[0] != a[1]
    assert len(a[0]) == 4


def test_no_failure_searches_full_horizon_instead_of_invalid_sentinel():
    p = np.zeros((200, 2))
    t = failure_timing(p, p, 20)
    assert not t['failed'] and t['failure_seconds'] is None
    windows = propose_windows(t, 201, 20, np.random.default_rng(5), {})
    assert windows[0]['end_seconds'] == 10
    assert len(windows) == 4
    with pytest.raises(ValueError, match='nonfinite'):
        failure_timing(p+np.nan, p, 20)


def test_prefix_gate_rejects_earlier_damage_even_with_better_late_motion():
    p = np.ones((200, 2))*.01; a = p.copy()
    before = trajectory_metrics(p, a, 20, 20)
    after_p = p.copy(); after_p[:20, 0] += .01
    after = trajectory_metrics(after_p, a, 20, 20)
    assert not prefix_gate(after, before)
    after_p = p.copy(); after_p[50:, 1] = .07
    after = trajectory_metrics(after_p, a, 20, 20)
    assert not prefix_gate(after, before)


def test_each_candidate_gets_its_own_window_and_full_hand_increment():
    from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
    from flash_chord.retargeting.local_correction import local_envelope, local_candidate
    base = np.zeros((201, 58), np.float32); base[:, 6] = 1; base[:, 35] = 1
    qids = list(range(3))+list(range(7,29))+list(range(29,32))+list(range(36,58))
    names = ['left_wrist_'+x for x in 'xyz']+['left_finger']*22+['right_wrist_'+x for x in 'xyz']+['right_finger']*22
    projector = NS(qids=qids, lower=np.full(50,-2.), upper=np.full(50,2.),
                   ball_joints=[(3,3),(32,31)], layout=NS(scalar_joints=NS(names=names)))
    e = NS(baseline=NS(joint_q=base, fps=20), binding=NS(joint_q=base.copy()), projector=projector,
           joint_basis=np.eye(58)[qids+[3,4,5,32,33,34]], options={}, window_ids=np.array([0,1]),
           window_contexts=[{'start_seconds':1.}, {'start_seconds':6.}],
           window_envelopes=np.stack([local_envelope(201,20,1,2), local_envelope(201,20,6,7)]))
    qs = []
    for i in range(2):
        AdaptiveWindowEditor.select_candidate_context(e,i)
        q = local_candidate(e,np.full(56,.5)); qs.append(q)
        np.testing.assert_array_equal(q[e.basis[:,0]==0], base[e.basis[:,0]==0])
        assert np.max(abs(q[:,7]-base[:,7])) > .01
        assert np.max(abs(q[:,36]-base[:,36])) > .01
    assert not np.array_equal(qs[0],qs[1])


def test_confirmation_requires_object_improvement_not_only_contact_gain():
    from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
    p = np.ones((20,2))*.01
    before = dict(all_errors_finite=True, full_horizon=True, object_position_robust_m=.05,
                  object_orientation_robust_rad=.2, object_success_robust_fraction=.5,
                  wrench_robust_mse=.4, unstable_world_fraction=0., per_object_position_mean_m=[.05,.05],
                  per_object_orientation_mean_rad=[.2,.2],contact_proximity_mean_m=.03,
                  per_object_contact_proximity_mean_m=[.03,.03],
                  per_hand_object_contact_proximity_mean_m=[[.03,None],[None,.03]],
                  **trajectory_metrics(p,p,20,5))
    after = copy.deepcopy(before); after['contact_proximity_mean_m'] = .02
    calls=[]
    def rollout(*args, **kwargs):
        calls.append(kwargs['initial_q_delta'].copy())
        return (before if len(calls)%2 else after),None,None
    reference = NS(joint_q=np.zeros((20,2)), num_joint_q=2, fps=20)
    e=NS(options={},baseline=reference,binding=reference,contact_proximity_weight=5.,protected_prefix_frames=5,
         projector=NS(qids=[0,1],layout=NS(scalar_joints=NS(names=['left_wrist_x','right_finger']))),rollout=rollout)
    e.acceptance_gate=lambda a,b: AdaptiveWindowEditor.acceptance_gate(e,a,b)
    assert not AdaptiveWindowEditor.confirm_candidate(e,reference,None,43,None,None)[0]
    after.update(object_position_robust_m=.04,per_object_position_mean_m=[.04,.04])
    assert AdaptiveWindowEditor.confirm_candidate(e,reference,None,43,None,None)[0]
    for i in range(0,len(calls),2): np.testing.assert_array_equal(calls[i],calls[i+1])


def test_window_specific_observations_and_confirmation_diversity(monkeypatch):
    from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
    from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
    import jax.numpy as jnp
    monkeypatch.setattr(TTSAReferenceEditor, 'observation', lambda *args: jnp.zeros((1,3)))
    e = AdaptiveWindowEditor.__new__(AdaptiveWindowEditor)
    e.baseline = NS(fps=20., num_frames=201)
    e.binding = NS(joint_q=np.zeros((201,58)))
    e.window_rng = np.random.default_rng(43); e.window_visits = {}
    e.upper_worlds=2048; e.timing_options={}; e.options={'confirmation_top_k':4}
    p=np.zeros((200,1,2));p[130:,0,1]=.03;p[160:,0,1]=.06
    obs=e.observation({}, {'frame_per_object_position_m':p,'frame_per_object_orientation_rad':p*0})
    assert obs.shape[0] == 2048
    assert np.unique(obs,axis=0).shape[0] == 4
    np.testing.assert_array_equal(np.bincount(e.window_ids),[512]*4)
    assert e.protected_prefix_frames == 130
    # Even when the globally cheapest proposals all belong to one window,
    # confirmation can inspect the other three locations.
    costs=np.arange(2048)/2048
    selected=e.shortlist_candidates(np.argsort(costs),np.ones(2048,bool),costs,2.)
    assert len(selected) == 4
    assert set(e.window_ids[selected]) == {0,1,2,3}
