"""Tests for the parts of the lab that can misrepresent an experiment."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .metrics import hand_metrics, rigid_align, trajectory_metrics
from .serve import confined
from .experiments import validate_result, BASE


class MetricsTests(unittest.TestCase):
    def test_rigid_alignment_does_not_remove_scale_error(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        r = Rotation.from_euler('z', 35, degrees=True).as_matrix()
        q = p @ r.T + [2, 3, 4]
        rr, t = rigid_align(p, q)
        np.testing.assert_allclose(p @ rr.T + t, q, atol=1e-10)
        rr, t = rigid_align(p * 2, q)
        self.assertGreater(np.linalg.norm(p * 2 @ rr.T + t - q), .5)

    def test_invalid_zero_pose_is_not_scored(self):
        p = np.repeat(np.eye(4)[None], 5, axis=0)
        p[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1]]
        q = p.copy(); q[:, :3, 3] += [2, 3, 4]; q[2] = 0
        result, _ = trajectory_metrics(p, q, [True, True, False, True, True])
        self.assertEqual(result['gt_frames'], 4)
        self.assertIsNone(result['object_error_mm'][2])
        self.assertLess(result['object_rigid_rmse_mm'], 1e-8)

    def test_outliers_are_flagged_not_silently_clamped(self):
        result = hand_metrics([[0, 0, .5], [0, 0, 12], [0, 0, 0]], [True, True, False], 20)
        self.assertEqual(result['far_frames'], [1])
        self.assertEqual(result['max_camera_distance_m'], 12)
        self.assertIsNone(result['camera_distance_m'][2])

    def test_file_server_rejects_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'site'; root.mkdir()
            (root / 'index.html').write_text('ok')
            (Path(d) / 'private').write_text('private')
            (root / 'link').symlink_to(Path(d) / 'private')
            self.assertIsNotNone(confined(root, 'index.html'))
            self.assertIsNone(confined(root, '../private'))
            self.assertIsNone(confined(root, 'link'))
            self.assertIsNone(confined(root, '.'))

    def test_registration_rejects_mislabeled_model_or_refinement(self):
        with tempfile.TemporaryDirectory() as d:
            result = Path(d) / 'result_slam_gravity_aligned'; result.mkdir()
            for name in ['result.npz', 'mesh.obj']:
                (result / name).write_bytes(b'fixture')
            manifest = result / 'manifest.json'
            manifest.write_text(json.dumps({'sources': {'hand_pose_source': 'hamer', 'hamer_crop_source': 'mask',
                                                         'final_hand_root': 'hamer_aligned_filled'}}))
            record = {'output_dir': d, 'config': {**BASE, 'hand': 'hawor'}}
            with self.assertRaisesRegex(ValueError, 'hand model'):
                validate_result(record)
            record['config'] = {**BASE, 'refine': 'on'}
            with self.assertRaisesRegex(ValueError, 'Refinement'):
                validate_result(record)
            record['config'] = {**BASE, 'crop': 'detector'}
            with self.assertRaisesRegex(ValueError, 'crop source'):
                validate_result(record)


if __name__ == '__main__':
    unittest.main()
