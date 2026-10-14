from types import SimpleNamespace as NS
import numpy as np
import pytest

from flash_chord.retargeting.failure_consensus import failure_consensus, perturb_initial_joints


def trajectories(events, *, objects=2, frames=220):
    p = np.zeros((frames, len(events), objects))
    for w, per_object in enumerate(events):
        for obj, frame in per_object.items():
            p[max(0, frame-10):frame, w, obj] = .03
            p[frame:, w, obj] = .06
    return p, p*0


def test_rare_early_failure_does_not_override_repeatable_lid_failure():
    p, a = trajectories([{0: 5}] + [{1: 60+i%5} for i in range(12)] + [{1: 160}]*3)
    t = failure_consensus(p, a, 20)
    assert t['selected_object_index'] == 1
    assert t['consensus_reached'] and t['consensus_fraction'] == 12/16
    assert t['failure_frame'] == 61 and t['onset_frame'] == 50
    assert t['selected_worlds'] == [1, 6, 11, 2, 7, 12, 3, 8, 4, 9, 5, 10]


def test_earliest_qualifying_cluster_precedes_larger_later_cluster():
    p, a = trajectories([{0: 40, 1: 140}]*8 + [{1: 140}]*8)
    t = failure_consensus(p, a, 20)
    assert t['selected_object_index'] == 0 and t['failure_frame'] == 40
    assert t['consensus_fraction'] == .5


def test_every_object_is_considered_even_if_another_failed_first():
    p, a = trajectories([{0: i*12, 1: 180} for i in range(16)])
    t = failure_consensus(p, a, 20)
    assert t['selected_object_index'] == 1 and t['failure_frame'] == 180
    assert t['consensus_fraction'] == 1.


def test_successful_worlds_stay_in_denominator_and_weak_agreement_is_explicit():
    p, a = trajectories([{1: 40}]*3 + [{}]*13)
    t = failure_consensus(p, a, 20)
    assert t['failed'] and not t['consensus_reached']
    assert t['selection_status'] == 'weak_consensus'
    assert t['consensus_fraction'] == 3/16 and t['baseline_failure_fraction'] == 3/16


def test_persistent_error_is_only_one_event_regardless_of_episode_length():
    p, a = trajectories([{1: 50}]*16, frames=400)
    short = failure_consensus(p[:100], a[:100], 20)
    long = failure_consensus(p, a, 20)
    assert short['failure_seconds'] == long['failure_seconds'] == 2.5
    assert short['selected_worlds'] == long['selected_worlds']


def test_clusters_cannot_chain_nearby_but_distant_endpoints():
    p, a = trajectories([{1: 20+4*i} for i in range(16)])
    t = failure_consensus(p, a, 20)
    assert not t['consensus_reached'] and t['consensus_fraction'] == 3/16
    assert t['failure_frame'] == 24


def test_orientation_failure_spikes_and_no_failure():
    p = np.zeros((100, 16, 1)); a = p.copy()
    p[5, :, 0] = 1.
    t = failure_consensus(p, a, 20)
    assert not t['failed'] and t['failure_seconds'] is None
    a[40:, :10, 0] = np.deg2rad(25)
    t = failure_consensus(p, a, 20)
    assert t['consensus_reached'] and t['failure_seconds'] == 2.
    assert t['consensus_fraction'] == 10/16


@pytest.mark.parametrize('frame', [0, 1, 5])
def test_initial_failure_still_produces_valid_early_window(frame):
    from flash_chord.retargeting.failure_windows import propose_windows
    p, a = trajectories([{0: frame}]*16)
    t = failure_consensus(p, a, 20)
    assert t['failure_frame'] == frame and t['onset_frame'] == 0
    windows = propose_windows(t, 221, 20, np.random.default_rng(0), {})
    assert len(windows) == 1 and windows[0]['end_seconds'] <= max(.05, frame/20)


def test_nonfinite_data_and_invalid_options_fail_loudly():
    p = np.zeros((30, 16, 2))
    with pytest.raises(ValueError, match='nonfinite'):
        failure_consensus(p+np.nan, p, 20)
    with pytest.raises(ValueError):
        failure_consensus(p, p, 20, minimum_fraction=0)
    with pytest.raises(ValueError):
        failure_consensus(p, p, 20, cluster_seconds=0)
    with pytest.raises(ValueError):
        failure_consensus(p[:, 0], p[:, 0], 20)


def test_reset_perturbations_reproducible_distinct_bounded_and_do_not_edit_reference():
    q = np.zeros(58, np.float32); q[[6, 35]] = 1
    names = ['left_wrist_x', 'left_finger', 'right_wrist_y', 'right_finger']
    ids = [0, 7, 30, 36]
    original = q.copy()
    args = q, names, ids, np.array([-.1, 0., -.1, -.2]), np.ones(4)*.2, 16, 43
    samples = perturb_initial_joints(*args)
    np.testing.assert_array_equal(samples, perturb_initial_joints(*args))
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(samples[:, [3, 4, 5, 6, 32, 33, 34, 35]],
                                  np.broadcast_to(q[[3, 4, 5, 6, 32, 33, 34, 35]], (16, 8)))
    assert len(np.unique(samples, axis=0)) == 16
    assert abs(samples[:, [0, 30]]).max() <= .00100001
    assert abs(samples[:, [7, 36]]).max() <= np.deg2rad(.50001)
    assert samples[:, 7].min() >= 0


def test_window_selection_uses_population_onsets_instead_of_nominal_failure(monkeypatch):
    from flash_chord.retargeting.adaptive_editor import AdaptiveWindowEditor
    from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
    import jax.numpy as jnp
    monkeypatch.setattr(TTSAReferenceEditor, 'observation', lambda *args: jnp.zeros((1, 3)))
    e = AdaptiveWindowEditor.__new__(AdaptiveWindowEditor)
    e.baseline = NS(fps=20., num_frames=221)
    e.binding = NS(joint_q=np.zeros((221, 58)))
    e.window_rng = np.random.default_rng(43); e.window_visits = {}
    e.upper_worlds = 2048; e.timing_options = {}; e.options = {}
    nominal, zero = trajectories([{0: 20}])
    p, a = trajectories([{1: 100}]*12+[{}]*4)
    observation = e.observation({}, dict(frame_per_object_position_m=nominal,
        frame_per_object_orientation_rad=zero, baseline_ensemble_position_m=p,
        baseline_ensemble_orientation_rad=a, baseline_ensemble_metadata={'baseline_probe_seed': 1}))
    assert e.failure_context['failure_seconds'] == 5.
    assert e.failure_context['consensus_fraction'] == .75
    assert e.protected_prefix_frames == 90
    assert observation.shape[0] == 2048
    np.testing.assert_array_equal(np.bincount(e.window_ids), [512]*4)
