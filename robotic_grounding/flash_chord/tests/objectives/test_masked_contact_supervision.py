"""CPU contact losses distinguish an unknown hand from observed no contact."""
import numpy as np
import pytest
import warp as wp

from flash_chord.objectives.contact import (
    contact_wrench_support_objective, missed_contact_penalty,
    unintended_contact_penalty, force_closure_objective,
)

KERNELS = [(contact_wrench_support_objective, (.1, .1)),
           (missed_contact_penalty, ()), (unintended_contact_penalty, ()),
           (force_closure_objective, (.1,))]


def launch(kernel, extra, command, current, valid, frames):
    command, current = np.asarray(command, np.float32), np.asarray(current, np.float32)
    worlds, hands, objects, basis = command.shape
    valid = np.asarray(valid, np.int32)
    with wp.ScopedDevice('cpu'):
        target = wp.array(command.ravel(), dtype=wp.float32)
        measured = wp.array(current.ravel(), dtype=wp.float32)
        output = wp.zeros(worlds, dtype=wp.float32)
        wp.launch(kernel, dim=worlds, inputs=[target, measured, hands, objects, basis,
            wp.array(frames, dtype=wp.int32), len(valid), wp.array(valid.ravel(), dtype=wp.int32),
            *extra], outputs=[output])
        np.testing.assert_array_equal(target.numpy(), command.ravel())
        np.testing.assert_array_equal(measured.numpy(), current.ravel())
        return output.numpy()


@pytest.mark.parametrize('kernel,extra', KERNELS)
def test_only_observed_hand_contributes_with_correct_per_frame_normalization(kernel, extra):
    command = np.array([[[[1, 1, 0], [0, 0, 0]], [[1, 0, 1], [0, 0, 0]]]], np.float32)
    current = np.array([[[[1, .5, 0], [.3, 0, 0]], [[0, 0, 1], [3, 2, 0]]]], np.float32)
    actual = launch(kernel, extra, np.repeat(command, 2, axis=0), np.repeat(current, 2, axis=0),
                    [[1, 0], [0, 1]], [0, 1])
    expected = [launch(kernel, extra, command[:, hand:hand+1], current[:, hand:hand+1],
                       [[1]], [0])[0] for hand in range(2)]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(launch(kernel, extra, command, current, [[0, 0]], [0]), [0.])


@pytest.mark.parametrize('kernel,extra,command,current,expected', [
    (contact_wrench_support_objective, (.1, .1), [1, 1, 0, 0], [1, .5, 0, 0], (1+np.exp(-1.6))/2),
    (missed_contact_penalty, (), [1, 1, 1, 0], [1, 0, 1, 0], 1/3),
    (unintended_contact_penalty, (), [0, 0, 0, 0], [.5, .3, 0, 0], 1+(.5**2+.3**2)/4),
    (force_closure_objective, (.1,), [1, 1, 1, 1], [1, .5, 0, 0], .5),
])
def test_all_valid_keeps_existing_analytic_contact_losses(kernel, extra, command, current, expected):
    actual = launch(kernel, extra, np.asarray(command).reshape(1,1,1,4),
                    np.asarray(current).reshape(1,1,1,4), [[1]], [0])
    np.testing.assert_allclose(actual, [expected], rtol=1e-6)
