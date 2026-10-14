from types import SimpleNamespace as NS

import numpy as np
import pytest

from flash_chord.retargeting.local_editor import scheduled_contact_context, confirmation_contact_gate
from flash_chord.retargeting.local_correction import local_envelope


def test_schedule_targets_roster_and_truncates_last_window():
    schedule = [dict(start_seconds=2., hand_index=0, object_index=0, focus='left_white_pot'),
                dict(start_seconds=12., hand_index=1, object_index=1, focus='right_wooden_spoon')]
    first = scheduled_contact_context(schedule, 1, 243, 20., (13, 2, 2))
    last = scheduled_contact_context(schedule, 2, 243, 20., (13, 2, 2))
    assert first['hand_index'] == 0 and first['end_seconds'] == 3.
    assert last['object_index'] == 1 and last['end_seconds'] == 12.1
    assert scheduled_contact_context(schedule, 3, 243, 20., (13, 2, 2)) == first
    weights = local_envelope(243, 20., last['start_seconds'], last['end_seconds'])
    assert weights[0] == weights[-1] == 0
    assert np.all(weights[:230] == 0)
    with pytest.raises(ValueError, match='outside the episode'):
        scheduled_contact_context(schedule, 2, 200, 20., (10, 2, 2))


def test_target_contact_gate_uses_indices_instead_of_object_name():
    before = dict(contact_proximity_mean_m=.1, per_object_contact_proximity_mean_m=[.1, .1],
                  per_hand_object_contact_proximity_mean_m=[[.1, None], [None, .1]])
    after = dict(contact_proximity_mean_m=.09, per_object_contact_proximity_mean_m=[.101, .08],
                 per_hand_object_contact_proximity_mean_m=[[.101, None], [None, .08]])
    e = NS(options={'contact_confirmation_tolerance_m': .01},
           local_context=dict(hand_index=1, object_index=1, focus='right_wooden_spoon'))
    assert confirmation_contact_gate(e, after, before)
    e.local_context = dict(hand_index=0, object_index=0, focus='left_white_pot')
    assert not confirmation_contact_gate(e, after, before)


def test_single_object_both_hands_can_target_same_spoon():
    schedule = [dict(start_seconds=1., hand_index=0, object_index=0, focus='left_wooden_spoon'),
                dict(start_seconds=6., hand_index=1, object_index=0, focus='right_wooden_spoon')]
    context = scheduled_contact_context(schedule, 2, 421, 20., (22, 2, 1))
    assert context['hand_index'] == 1 and context['object_index'] == 0
    before = dict(contact_proximity_mean_m=.1, per_object_contact_proximity_mean_m=[.1],
                  per_hand_object_contact_proximity_mean_m=[[.1], [.1]])
    after = dict(contact_proximity_mean_m=.09, per_object_contact_proximity_mean_m=[.09],
                 per_hand_object_contact_proximity_mean_m=[[.105], [.075]])
    assert confirmation_contact_gate(NS(options={'contact_confirmation_tolerance_m': .01}, local_context=context), after, before)
