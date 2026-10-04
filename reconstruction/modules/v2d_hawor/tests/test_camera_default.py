"""Small CPU checks for the camera-only HaWoR integration."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

RECON = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hawor = _load(RECON / "modules/v2d_hawor/lib/run_hawor.py", "hawor_camera_local")
sys.path.insert(0, str(RECON / "modules/v2d_pipelines"))
import run_ego_wilor as pipeline  # noqa: E402
import run_ego_reconstruction as entrypoint  # noqa: E402


class CameraDefaultTest(unittest.TestCase):
    def test_runner_calls_only_detector_and_camera_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"smoke")
            seq = root / "sequence"
            seq.mkdir()
            calls = []
            demo = types.ModuleType("demo")
            demo.detect_track_video = lambda args: (calls.append("detect") or
                                                    (0, 4, str(seq), []))
            demo.hawor_motion_estimation = lambda *args: (calls.append("motion") or
                                                          ([], 900.0))
            demo.hawor_slam = lambda *args: self.fail("HaWoR SLAM was called")
            demo.hawor_infiller = lambda *args: self.fail("HaWoR infiller was called")
            with patch.dict(sys.modules, {"demo": demo}), patch.object(
                hawor, "_HAWOR_ROOT", str(root)
            ), patch.object(hawor, "_prepare_weight_links"), patch.object(
                hawor, "_patch_renderer_mask_fallback"
            ), patch.object(hawor, "_video_frame_count", return_value=3), patch.object(
                hawor, "_video_fps", return_value=20.0
            ), patch.object(hawor, "_motion_cache_current", return_value=True), patch.object(
                hawor, "_write_motion_cache_meta"
            ), patch.object(hawor, "_export_camera_space_tracks") as export:
                hawor.run_hawor(str(video), str(root / "tracks"), str(root / "weights"),
                                native_dir=str(root / "native"), left_id=-1, right_id=7)
                self.assertEqual(calls, ["detect", "motion"])
                self.assertEqual(export.call_args.args[-1], 20.0)

    def test_export_keeps_source_timeline_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "native"
            right = seq / "cam_space/1"
            right.mkdir(parents=True)
            eye = np.eye(3).tolist()
            (right / "0_3.json").write_text(json.dumps({
                "init_root_orient": [[eye] * 4],
                "init_hand_pose": [[[eye] * 15 for _ in range(4)]],
                "init_trans": [[[0.1, 0.2, 0.5]] * 4],
                "init_betas": [[[0.0] * 10] * 4],
            }))
            out = root / "tracks"
            (out / "99").mkdir(parents=True)
            (out / "99/000000.json").write_text("{}")
            boxes = {i: {"x0": 1.0, "y0": 1.0, "x1": 3.0, "y1": 3.0}
                     for i in (0, 3)}
            with patch.object(hawor, "_image_size", return_value=(4, 4)), patch.object(
                hawor, "_bbox_lookup", return_value={False: {}, True: boxes}
            ):
                hawor._export_camera_space_tracks(
                    str(seq), str(out), 900.0, -1, 4, 0, 4, 3,
                    source_fps=20.0,
                )
            self.assertFalse((out / "99").exists())
            self.assertEqual([p.name for p in (out / "4").glob("*.json")],
                             ["000000.json", "000002.json"])
            record = json.loads((out / "4/000002.json").read_text())
            self.assertEqual((record["frame_idx"], record["hawor_native_frame_idx"]), (2, 3))
            self.assertEqual(record["time_s"], 0.1)
            self.assertEqual((record["track_id"], record["is_right"]), (4, True))
            meta = json.loads((out / "_metadata.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["coverage"]["right"]["missing_frames"]["no_detection_bbox"], [1])
            self.assertEqual(meta["source_fps"], 20.0)

    def test_hand_ids_are_from_hand_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hand_tracks.json"
            path.write_text(json.dumps({"tracks": [
                {"object_id": 1, "role": "object", "is_right": False},
                {"object_id": 7, "role": "hand", "is_right": True},
            ]}))
            self.assertEqual(pipeline._hawor_hand_track_ids(str(path)), (-1, 7))
            path.write_text(json.dumps({"tracks": [
                {"object_id": 4, "role": "hand", "is_right": False},
                {"object_id": 7, "role": "hand", "is_right": True},
            ]}))
            self.assertEqual(pipeline._hawor_hand_track_ids(str(path)), (4, 7))

    def test_bbox_lookup_excludes_interpolated_detector_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp)
            tracks = seq / "tracks_0_3"
            tracks.mkdir()
            np.save(tracks / "model_tracks.npy", {1: [
                {"frame": 0, "det": True, "det_handedness": [1],
                 "det_box": [2, 2, 4, 4]},
                {"frame": 1, "det": False, "det_handedness": [1],
                 "det_box": [0, 0, 0, 0]},
                {"frame": 2, "det": True, "det_handedness": [1],
                 "det_box": [3, 3, 4, 4]},
            ]})
            boxes = hawor._bbox_lookup(str(seq), 0, 3)
            self.assertEqual(set(boxes[True]), {0, 2})
            self.assertEqual(boxes[False], {})

    def test_missing_detector_output_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                hawor._bbox_lookup(tmp, 0, 3)

    def test_detector_miss_is_a_completed_sparse_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "7").mkdir()
            (root / "_metadata.json").write_text(json.dumps({
                "export_version": 6, "status": "complete",
                "left_conversion": "mirror_conjugate_axis_angle_yz",
                "focal_length": 900.0, "left_id": -1, "right_id": 7,
                "source_frame_count": 3,
            }))
            self.assertTrue(pipeline._hawor_outputs_current(str(root), 900.0, -1, 7, 3))
            self.assertFalse(pipeline._hawor_outputs_current(str(root), 900.0, -1, 7, 4))

    def test_entrypoint_defaults_to_hawor(self):
        with patch.object(sys, "argv", ["run_ego_reconstruction.py", "--video", "x.mp4",
                                        "--object_prompt", "pot", "--output_dir", "out"]):
            self.assertEqual(entrypoint.parse_args().hand_tracking, "hawor")
        with patch.object(sys, "argv", ["run_ego_reconstruction.py", "--video", "x.mp4",
                                        "--object_prompt", "pot", "--output_dir", "out",
                                        "--hand_tracking", "hamer"]):
            self.assertEqual(entrypoint.parse_args().hand_tracking, "hamer")


if __name__ == "__main__":
    unittest.main()
