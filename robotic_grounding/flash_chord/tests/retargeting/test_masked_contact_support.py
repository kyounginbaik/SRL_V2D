"""Upper cost and acceptance use the same observed-hand support contract."""
import numpy as np
import pytest

from flash_chord.retargeting.chord_upper import WrenchSupportAccumulator
from flash_chord.retargeting.local_acceptance import LocalWrenchAccumulator
from flash_chord.retargeting.wrench_hypergradient_editor import support_cost


@pytest.mark.parametrize('observed_hand', [0, 1])
def test_unknown_hand_cannot_change_upper_cost_or_active_inactive_denominators(observed_hand):
    reference = np.array([[[[1., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]])
    current = np.array([[[[.3, 0.], [1., 2.]], [[3., 2.], [1., 0.]]]])
    before = reference.copy()
    valid = np.array([[observed_hand == 0, observed_hand == 1]])
    only_reference = reference[:, observed_hand:observed_hand+1]
    only_current = current[:, observed_hand:observed_hand+1]
    np.testing.assert_allclose(support_cost(current, reference, hand_valid=valid),
                               support_cost(only_current, only_reference))
    masked = WrenchSupportAccumulator(1, 2, 2, 2)
    single = WrenchSupportAccumulator(1, 1, 2, 2)
    masked.add(current, reference, hand_valid=valid)
    single.add(only_current, only_reference)
    assert masked.summaries() == single.summaries()
    np.testing.assert_array_equal(reference, before)
    # A measured no-contact target on a valid hand must still reject extra contact.
    assert masked.summaries()[0]['cws_unintended_energy'] > 0


def test_default_all_valid_and_unknown_local_frames():
    reference = np.zeros((1,2,1,2))
    current = np.ones_like(reference)
    np.testing.assert_array_equal(support_cost(current, reference),
        support_cost(current, reference, hand_valid=[[1,1]]))
    np.testing.assert_array_equal(support_cost(current, reference, hand_valid=[[0,0]]), [0.])
    legacy, explicit = WrenchSupportAccumulator(1,2,1,2), WrenchSupportAccumulator(1,2,1,2)
    legacy.add(current, reference)
    explicit.add(current, reference, hand_valid=[[1,1]])
    assert legacy.summaries() == explicit.summaries()
    local = LocalWrenchAccumulator(1,2,1,2, region=dict(start_frame=1, stop_frame_exclusive=2))
    local.add(current, reference, hand_valid=[[1,1]])
    local.add(current*100, reference, hand_valid=[[0,0]])
    summary = local.summaries()[0]
    assert summary['cws_unintended_energy'] == 1.
    assert summary['local_cws_unintended_energy'] == 0.
    assert summary['local_cws_frames'] == 1


@pytest.mark.parametrize('mask', [[[np.nan,1]], [[-1,1]], [1,1]])
def test_invalid_hand_mask_is_rejected(mask):
    reference = np.ones((1,2,1,2))
    with pytest.raises(ValueError, match='hand validity'):
        support_cost(reference, reference, hand_valid=mask)
    with pytest.raises(ValueError, match='hand validity'):
        WrenchSupportAccumulator(1,2,1,2).add(reference, reference, hand_valid=mask)
