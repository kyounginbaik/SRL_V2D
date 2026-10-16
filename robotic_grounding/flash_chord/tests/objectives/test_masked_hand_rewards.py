"""CPU reduction excludes invalid hand labels without altering object rewards."""
import numpy as np
import warp as wp

from flash_chord.objectives.tracking import normalize_tracking_objectives


def reduce(mask):
    with wp.ScopedDevice("cpu"):
        i32=lambda x:wp.array(x,dtype=wp.int32)
        f32=lambda x:wp.array(x,dtype=wp.float32)
        objects=f32([4.,6.,8.]);keypoints=wp.zeros(3);fingers=wp.zeros(3)
        wp.launch(normalize_tracking_objectives,dim=3,inputs=[2,6,2,1,1,1,
            f32([6.,600.,600.,3.,600.,600.]),f32([.8,900.,900.,.4,900.,900.]),
            i32([0,1,2]),3,i32(np.asarray(mask).reshape(-1))],outputs=[objects,keypoints,fingers])
        return objects.numpy().copy(),keypoints.numpy().copy(),fingers.numpy().copy()


def test_only_observed_hands_contribute_and_empty_frames_have_zero_hand_reward():
    objects,keypoints,fingers=reduce([[1,0],[0,1],[0,0]])
    np.testing.assert_allclose(objects,[2.,3.,4.])
    np.testing.assert_allclose(keypoints,[1.,.5,0.])
    np.testing.assert_allclose(fingers,[.8,.4,0.])


def test_all_valid_reduction_preserves_previous_hand_average():
    _,keypoints,fingers=reduce(np.ones((3,2),int))
    np.testing.assert_allclose(keypoints,[50.5,50.25,100.])
    np.testing.assert_allclose(fingers,[450.4,450.2,900.])
