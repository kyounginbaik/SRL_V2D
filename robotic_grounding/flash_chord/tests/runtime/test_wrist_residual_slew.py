"""CPU Warp checks for bounded residual targets, independent of physics/GPU."""
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from flash_chord.runtime.actions import ActionBuffers, ActionConfig
from test_action import _layout, _reference


def make_buffers(*, fps=20., limited=True, worlds=2):
    config = ActionConfig(input_mode="normalized", normalized_mapping="rational",
                          wrist_pos_clip=.05, wrist_ori_clip=.5, ema=.3)
    if limited:
        config = replace(config, wrist_residual_linear_speed_limit_m_s=.15,
                         wrist_residual_angular_speed_limit_rad_s=1.5)
    return ActionBuffers.build(_layout(), replace(_reference(), fps=fps), worlds,
                               config=config, device="cpu")


def process(buffers, actions, frames=None):
    actions = np.asarray(actions, np.float32).reshape(buffers.world_count, -1)
    frames = np.zeros(buffers.world_count, np.int32) if frames is None else np.asarray(frames, np.int32)
    buffers.process(wp.array(actions.ravel(), dtype=wp.float32, device="cpu"),
                    wp.array(frames, dtype=wp.int32, device="cpu"))
    return buffers.filtered_action.numpy().reshape(buffers.world_count, -1)


@pytest.mark.parametrize("fps", [20., 40.])
def test_extreme_changes_are_limited_per_hand_norm_and_reference_is_not_slowed(fps):
    limited, unrestricted = make_buffers(fps=fps), make_buffers(fps=fps, limited=False)
    previous = np.zeros((2, 16), np.float32)
    for sign in (1., 1., -1., -1., 1.):
        # Hands and worlds request different directions; limits apply separately.
        actions = np.array([[sign] * 8 + [-sign] * 8,
                            [-sign] * 8 + [sign] * 8], np.float32)
        actual = process(limited, actions, [0, 1])
        unbounded = process(unrestricted, actions, [0, 1])
        for start in (0, 8):
            for offset, speed in ((0, .15), (3, 1.5)):
                delta = actual[:, start+offset:start+offset+3] - previous[:, start+offset:start+offset+3]
                np.testing.assert_allclose(np.linalg.norm(delta, axis=1), speed/fps, rtol=2e-6, atol=1e-8)
            # Finger exploration and its range are unaffected by wrist limits.
            np.testing.assert_array_equal(actual[:, start+6:start+8], unbounded[:, start+6:start+8])
        targets = limited.processed_target.numpy().reshape(2, 18)
        np.testing.assert_allclose(targets[:, :3]-actual[:, :3], [[1.,2.,3.],[4.,5.,6.]], atol=5e-7)
        np.testing.assert_allclose(targets[:, 9:12]-actual[:, 8:11], [[-1.,-2.,-3.],[-4.,-5.,-6.]], atol=5e-7)
        previous = actual.copy()


def test_nonzero_history_zero_action_and_selected_world_reset():
    buffers = make_buffers()
    previous = np.array([[.02,-.01,.03,.2,-.1,.15,.3,-.4,
                          -.03,.02,-.01,-.2,.15,-.1,-.2,.4]] * 2, np.float32)
    buffers.filtered_action.assign(previous.ravel())
    actual = process(buffers, np.zeros((2,16)))
    for start in (0,8):
        for offset, limit in ((0,.15/20), (3,1.5/20)):
            before = previous[:, start+offset:start+offset+3]
            after = actual[:, start+offset:start+offset+3]
            delta = -.7*before
            expected = before + delta*np.minimum(1., limit/np.linalg.norm(delta,axis=1))[:,None]
            np.testing.assert_allclose(after,expected,rtol=2e-6,atol=1e-8)
    retained = actual[1].copy()
    buffers.reset(wp.array([1,0],dtype=wp.int32,device="cpu"))
    reset = buffers.filtered_action.numpy().reshape(2,16)
    np.testing.assert_array_equal(reset[0],np.zeros(16))
    np.testing.assert_array_equal(reset[1],retained)
    zero = process(buffers,np.zeros((2,16)))
    np.testing.assert_array_equal(zero[0],np.zeros(16))
    np.testing.assert_array_equal(buffers.processed_target.numpy().reshape(2,18)[0,:3],[1.,2.,3.])


def test_disabled_defaults_preserve_legacy_filter_and_regularization():
    buffers = make_buffers(limited=False)
    assert buffers.wrist_linear_step_limit == buffers.wrist_angular_step_limit == 0.
    rng = np.random.default_rng(7)
    expected = np.zeros((2,16),np.float32)
    previous_raw = np.zeros_like(expected)
    scale, clip, mapping_scale = (getattr(buffers,k).numpy() for k in ("scale","clip","input_scale"))
    for actions in (rng.uniform(-1,1,(2,16)).astype(np.float32), np.ones((2,16),np.float32), -np.ones((2,16),np.float32)):
        magnitude = np.abs(actions)
        raw = actions/((1-magnitude)+magnitude/mapping_scale)
        expected = np.clip(.3*expected+.7*raw*scale,-clip,clip)
        actual = process(buffers,actions)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-8)
        np.testing.assert_allclose(buffers.action_l2.numpy(),np.sum(raw*raw,axis=1),rtol=2e-6)
        np.testing.assert_allclose(buffers.action_rate_l2.numpy(),np.sum((raw-previous_raw)**2,axis=1),rtol=2e-6)
        previous_raw = raw
    # Existing normalized rational endpoints still reach the configured clip.
    assert np.max(np.abs(actual[:,:3])) > .15/20


@pytest.mark.parametrize("name", ["wrist_residual_linear_speed_limit_m_s", "wrist_residual_angular_speed_limit_rad_s"])
@pytest.mark.parametrize("value", [0.,-1.,float("nan"),float("inf"),True])
def test_invalid_config_is_rejected(name,value):
    with pytest.raises(ValueError,match=name):
        ActionBuffers.build(_layout(),_reference(),1,
                            config=replace(ActionConfig(),**{name:value}),device="cpu")
