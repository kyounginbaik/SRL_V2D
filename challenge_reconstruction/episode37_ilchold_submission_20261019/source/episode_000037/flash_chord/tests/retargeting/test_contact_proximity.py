"""Contact reach must follow each object's actual pose and intended hand."""

import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_proximity import ContactProximity, summarize_contact_proximity


class TwoObjectReference:
    def object_body_names(self):
        return ["spoon", "rack"]

    def object_body_pos_w(self):
        return np.array([[[0., 0., 0.], [1., 0., 0.]]])

    def object_body_quat_w(self):
        return np.array([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])

    def contact_part_ids(self, side):
        return np.array([[1, 0]]) if side == "left" else np.array([[2, 0]])

    def contact_pos_w(self, side):
        return (np.array([[[.1, 0., 0.], [0., 0., 0.]]]) if side == "left"
                else np.array([[[1., .1, 0.], [0., 0., 0.]]]))


def test_proximity_tracks_actual_objects_and_intended_hands():
    evaluator = ContactProximity(TwoObjectReference(), ("left", "right"))
    poses = np.array([[[2., 0., 0., 0., 0., 0., 1.],
                       [0., 2., 0., 0., 0., 0., 1.]]])
    tips = np.array([[[[2.1, 0., 0.]], [[0., 2.1, 0.]]]])
    sums, counts = evaluator.frame([0], tips, poses)
    result = summarize_contact_proximity(sums, counts)
    assert result["contact_proximity_mean_m"] < 1e-8
    assert result["per_object_contact_proximity_mean_m"] == [0., 0.]
    assert result["per_hand_object_contact_proximity_mean_m"] == [[0., None], [None, 0.]]
    # Moving only the spoon-side fingertips changes only that object's score.
    tips[0, 0, 0, 0] += .04
    changed = summarize_contact_proximity(*evaluator.frame([0], tips, poses))
    np.testing.assert_allclose(changed["per_object_contact_proximity_mean_m"], [.04, 0.], atol=1e-8)
    _, _, vectors = evaluator.frame([0], tips, poses, return_vectors=True)
    np.testing.assert_allclose(vectors[0, 0, 0], [-.04, 0., 0.], atol=1e-8)
    np.testing.assert_allclose(vectors[0, 1, 1], [0., 0., 0.], atol=1e-8)


def test_proximity_respects_object_rotation_and_caps_world_outliers():
    evaluator = ContactProximity(TwoObjectReference(), ("left", "right"))
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    poses = np.array([[[2., 0., 0., *q], [0., 2., 0., 0., 0., 0., 1.]]])
    tips = np.array([[[[2., .1, 0.]], [[0., 2.1, 0.]]]])
    summary = summarize_contact_proximity(*evaluator.frame([0], tips, poses))
    assert summary["contact_proximity_mean_m"] < 1e-8
    summary = summarize_contact_proximity(np.array([[[0., 10.]], [[0., 0.]]]),
                                          np.ones((2, 1, 2), dtype=int))
    np.testing.assert_allclose(summary["per_object_contact_proximity_mean_m"], [0., 1.])
    poses[0, 0, 3:] = 0.
    invalid = summarize_contact_proximity(*evaluator.frame([0], tips, poses))
    assert invalid["per_object_contact_proximity_mean_m"][0] == 2.


def test_far_hand_does_not_hide_reachable_hand_ranking():
    counts = np.ones((1, 2, 1), dtype=int)
    closer = summarize_contact_proximity(np.array([[[2.], [.4]]]), counts)
    farther = summarize_contact_proximity(np.array([[[2.], [.44]]]), counts)
    assert np.isclose(closer["contact_proximity_mean_m"], 1.2)
    assert farther["contact_proximity_mean_m"] > closer["contact_proximity_mean_m"]
