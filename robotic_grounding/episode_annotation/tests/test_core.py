"""Offline contract checks for episode annotation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from episode_annotation.core import Episode, sample_frames, validate_annotations
from episode_annotation.render import _font, render


def fixture_data(episode: Episode) -> dict:
    return {
        "schema_version": 1, "provenance": {"prompt_version": "offline-fixture", "calls": []},
        "episode": {**episode.metadata(), "model_requested": "gpt-6.1-sol", "actual_models": ["gpt-6.1-sol"],
                    "granularity": "mid", "contact": "on", "sampling": {}},
        "objects": [{"id": "box", "description": "small box"}], "task_summary": "Lift a box",
        "stages": [
            {"id": "S01", "name": "Approach", "description": "The hand approaches the box",
             "start_frame": 0, "end_frame": 2, "start_time": 0.0, "end_time_exclusive": 0.2,
             "end_time_is_estimated": False, "confidence": "high", "hand_roles": {"left": "approaches", "right": "idle"}, "related_object_ids": ["box"],
             "relevant_pairs": [{"hand": "left", "object_id": "box"}], "key_frame_ids": [0]},
            {"id": "S02", "name": "Lift", "description": "The hand lifts the box",
             "start_frame": 2, "end_frame": 3, "start_time": 0.2, "end_time_exclusive": 0.3,
             "end_time_is_estimated": True, "confidence": "medium", "hand_roles": {"left": "lifts", "right": "idle"}, "related_object_ids": ["box"],
             "relevant_pairs": [{"hand": "left", "object_id": "box"}], "key_frame_ids": [2]},
        ],
        "boundaries": [{"estimated_start_frame": 2, "last_observed_before": 1,
                        "first_observed_after": 2, "confidence": "medium", "note": "Change between frames"}],
        "observed_frame_ids": [0, 1, 2],
        "contacts": [{"frame_id": 0, "hand": "left", "object_id": "box", "value": 0,
                      "evidence": "A visible gap remains", "confidence": "high"},
                     {"frame_id": 2, "hand": "left", "object_id": "box", "value": None,
                      "evidence": "The edge is occluded", "confidence": "low"}],
    }


class EpisodeTests(unittest.TestCase):
    def test_render_font_scales(self) -> None:
        small = _font(12).getbbox("Contact labels")
        large = _font(24).getbbox("Contact labels")
        self.assertGreater(large[2] - large[0], small[2] - small[0])

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        for name, color in (("frame10.png", "red"), ("frame2.png", "blue"), ("frame1.png", "green")):
            Image.new("RGB", (80, 40), color).save(self.frames / name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_frame_order_and_timestamp_mapping(self) -> None:
        stamp = self.root / "times.json"
        stamp.write_text(json.dumps({"frame1.png": 0.0, "frame2.png": 0.15, "frame10.png": 0.9}))
        episode = Episode(self.frames, timestamps=stamp)
        self.assertEqual([x.name for x in episode.files], ["frame1.png", "frame2.png", "frame10.png"])
        self.assertEqual(episode.times, [0.0, 0.15, 0.9])
        self.assertEqual(sample_frames(episode.times, 2), [0, 2])
        self.assertTrue(episode.frame(1, self.root / "cache").exists())

    def test_fps_mapping_and_endpoints(self) -> None:
        episode = Episode(self.frames, fps=10)
        self.assertEqual(episode.times, [0, 0.1, 0.2])
        self.assertEqual(sample_frames(episode.times, 1), [0, 2])
        with self.assertRaises(ValueError):
            Episode(self.frames)

    def test_video_source_frame_mapping(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv-python and numpy are required for video decoding")
        path = self.root / "synthetic.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (80, 40))
        if not writer.isOpened():
            self.skipTest("MP4 encoder is unavailable")
        for value in (20, 80, 140, 200):
            writer.write(np.full((40, 80, 3), value, dtype=np.uint8))
        writer.release()
        episode = Episode(path)
        self.assertEqual(episode.frame_count, 4)
        self.assertEqual(Episode(path).identity, episode.identity)
        self.assertEqual(sample_frames(episode.times, 2), [0, 3])
        extracted = episode.frame(2, self.root / "video_cache")
        with Image.open(extracted) as image:
            self.assertAlmostEqual(image.getpixel((20, 20))[0], 140, delta=10)

    def test_interval_contact_and_render(self) -> None:
        episode = Episode(self.frames, fps=10)
        data = fixture_data(episode)
        validate_annotations(data)
        path = self.root / "annotations.json"
        path.write_text(json.dumps(data))
        outputs = render(path)
        self.assertTrue((self.root / "stage_contact_sheet.png").exists())
        self.assertTrue((self.root / "stage_timeline.png").exists())
        self.assertEqual(len(outputs), 4)
        data["stages"][1]["start_frame"] = 1
        with self.assertRaises(ValueError):
            validate_annotations(data)

    def test_invalid_contact_and_disabled_mode(self) -> None:
        data = fixture_data(Episode(self.frames, fps=10))
        data["contacts"][1]["value"] = 2
        with self.assertRaises(ValueError):
            validate_annotations(data)
        data["episode"]["contact"] = "off"
        data["contacts"] = []
        validate_annotations(data)


if __name__ == "__main__":
    unittest.main()
