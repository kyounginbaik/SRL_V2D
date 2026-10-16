from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from flash_chord.retargeting.local_acceptance import (
    scoring_region, local_pose_metrics, LocalWrenchAccumulator, local_guard_reasons,
)
from flash_chord.retargeting.failure_windows import trajectory_metrics
from flash_chord.retargeting.response_editor import response_cost


def test_failure_region_includes_lift_after_edit_and_clips_at_episode_end():
    timing = dict(failed=True, onset_frame=22, failure_seconds=1.65)
    region = scoring_region(timing, 335, 20.)
    assert (region['start_frame'], region['stop_frame_exclusive']) == (22, 53)
    assert scoring_region(timing, 40, 20.)['stop_frame_exclusive'] == 39
    assert scoring_region(dict(failed=False), 335, 20.)['start_frame'] == 0
    assert scoring_region(dict(failed=False), 335, 20.)['stop_frame_exclusive'] == 334


def test_initial_frame_failure_still_has_nonempty_scoring_interval():
    region=scoring_region(dict(failed=True,onset_frame=0,failure_seconds=0.),3,20.)
    assert region['start_frame']==0 and region['stop_frame_exclusive']==2


def test_local_wrench_uses_only_shared_interval_and_keeps_full_diagnostics():
    accumulator=LocalWrenchAccumulator(2,1,2,1,region=dict(start_frame=1,stop_frame_exclusive=3))
    reference=np.ones((2,1,2,1))
    for frame in range(4):
        current=reference.copy()
        current[0]=0 if frame in (0,3) else 1
        current[1]=0 if frame in (1,2) else 1
        accumulator.add(current,reference)
    a,b=accumulator.summaries()
    assert a['cws_frames']==4 and a['local_cws_frames']==2
    assert a['local_cws_missed_fraction']==0 and b['local_cws_missed_fraction']==1
    assert a['cws_missed_fraction']==b['cws_missed_fraction']==.5


def metrics(p, a, region):
    return dict(local_pose_metrics(p,a,region), **trajectory_metrics(p,a,20.,region['start_frame']),
        all_errors_finite=True, full_horizon=True, unstable_world_fraction=0.,
        per_object_position_mean_m=p.mean(0).tolist(), per_object_orientation_mean_rad=a.mean(0).tolist())


def example():
    region=dict(start_frame=10,stop_frame_exclusive=30)
    p=np.full((100,2),.02); p[10:30,1]=.12
    a=np.zeros_like(p)
    return p,a,region,metrics(p,a,region)


def test_local_improvement_survives_small_later_regression_and_wrench_submetric_change():
    p,a,r,b=example()
    p[10:30,1]=.04; p[30:,0]+=.005
    c=metrics(p,a,r)
    # These old vetoes are deliberately absent; aggregate local objective decides.
    c['per_object_cws_missed_fraction']=[.03,.1]
    b['per_object_cws_missed_fraction']=[0.,.1]
    c['per_object_first_failure_seconds']=[2.,1.]
    b['per_object_first_failure_seconds']=[5.,1.]
    assert local_guard_reasons(c,b)==[]
    assert c['local_object_position_robust_m'] < b['local_object_position_robust_m']


@pytest.mark.parametrize('where', ['prefix','tail'])
def test_better_lid_cannot_hide_large_damage_to_pot_elsewhere(where):
    p,a,r,b=example()
    p[10:30,1]=.01
    p[:10 if where=='prefix' else 0,0]+=.02
    if where=='tail': p[30:,0]+=.08
    failures=local_guard_reasons(metrics(p,a,r),b)
    assert ('prefix_position_mean_m' if where=='prefix' else 'tail_position_mean_m') in failures


def test_each_object_protected_even_if_other_object_improves():
    p,a,r,b=example()
    c=deepcopy(b)
    c['per_object_position_mean_m']=[b['per_object_position_mean_m'][0]+.06,0.]
    assert 'per_object_position_mean_m' in local_guard_reasons(c,b)


def test_mismatched_horizons_and_nan_fail_closed():
    p,a,r,b=example()
    c=deepcopy(b);c['local_start_frame']+=1
    assert 'local_start_frame_mismatch' in local_guard_reasons(c,b)
    c=deepcopy(b);c['tail_position_mean_m'][0]=np.nan
    assert 'tail_position_mean_m' in local_guard_reasons(c,b)
    p[20,1]=np.nan
    with pytest.raises(FloatingPointError): local_pose_metrics(p,a,r)


def test_local_objective_is_not_diluted_by_long_unrelated_episode_tail():
    q=np.zeros((100,58));q[:,[6,35]]=1.
    layout=NS(hands=[NS(finger_q_ids=np.arange(7,29)),NS(finger_q_ids=np.arange(36,58))])
    p,a,r,b=example()
    p[10:30,1]=.04
    c=metrics(p,a,r)
    for m in [b,c]:
        for name in ['cws_tolerance_error','cws_missed_fraction','cws_unintended_fraction','cws_unintended_energy']:
            m['local_'+name]=0.
    before=response_cost(b,q,q,layout,20.,local=True)
    after=response_cost(c,q,q,layout,20.,local=True)
    assert after<before
    # Irrelevant whole-episode summaries cannot dominate candidate ranking.
    c.update(object_position_robust_m=.5,object_orientation_robust_rad=2.,cws_tolerance_error=9.)
    assert response_cost(c,q,q,layout,20.,local=True)==after
