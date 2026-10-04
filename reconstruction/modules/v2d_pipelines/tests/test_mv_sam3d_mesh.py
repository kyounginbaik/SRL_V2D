"""CPU checks for GT-free view selection and model input pairing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv_sam3d_mesh import (  # noqa: E402
    MV_SAM3D_COMMIT, prepare_views, record_single_view, run_mv_sam3d,
)


class ViewSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames = self.root / "frames"
        self.depth = self.root / "depth"
        self.masks = self.root / "masks" / "7"
        for path in (self.frames, self.depth, self.masks):
            path.mkdir(parents=True)
        (self.root / "object_track.json").write_text(
            json.dumps({"object_id": 7, "prompt": "red pan", "reference_frame": 17}))
        for index in (3, 9, 17, 24, 50, 81):
            name = f"{index:06d}.png"
            Image.fromarray(np.full((24, 32, 3), index, np.uint8)).save(self.frames / name)
            Image.fromarray(np.full((24, 32), 1, np.uint8)).save(self.depth / name)
            mask = np.zeros((24, 32), np.uint8)
            mask[5:20, 6:26] = 255
            Image.fromarray(mask).save(self.masks / name)

    def tearDown(self):
        self.tmp.cleanup()

    def select(self, good):
        def fake_inspect(image, mask, depth, hand_masks):
            index = int(image.stem)
            return {"frame": index, "passes_quality": index in good,
                    "rejection_reasons": [] if index in good else ["quality_gate"]}
        with patch("mv_sam3d_mesh.inspect_view", side_effect=fake_inspect):
            return prepare_views(
                frames_dir=str(self.frames), masks_dir=str(self.root / "masks"),
                depth_dir=str(self.depth), object_id=7, reference_frame=17,
                output_dir=str(self.root / "mesh_mv"), seed=29,
                object_prompt="red pan")

    def test_four_views_include_reference_and_temporal_endpoints(self):
        manifest = self.select({3, 9, 17, 24, 50, 81})
        self.assertEqual(manifest["selected_frames"], [17, 3, 24, 81])
        self.assertEqual(manifest["actual_backend"], "mvsam3d")
        self.assertEqual(manifest["seed"], 29)
        self.assertEqual(len(manifest["views"]), 4)
        for view in manifest["views"]:
            index = view["view_index"]
            input_root = Path(manifest["input_root"])
            rgb = np.asarray(Image.open(input_root / "images" / f"{index}.png"))
            rgba = np.asarray(Image.open(input_root / "object" / f"{index}.png"))
            self.assertTrue(np.array_equal(rgb, rgba[..., :3]))
            self.assertTrue(np.array_equal(rgba[..., 3] > 0,
                np.asarray(Image.open(view["mask_source"])) > 0))

    def test_two_or_three_views_are_kept(self):
        for good, expected in [({3, 17}, [17, 3]), ({3, 17, 81}, [17, 3, 81])]:
            with self.subTest(good=good):
                manifest = self.select(good)
                self.assertEqual(manifest["selected_frames"], expected)
                self.assertEqual(manifest["actual_backend"], "mvsam3d")

    def test_one_view_falls_back_with_reason(self):
        manifest = self.select({24})
        self.assertEqual(manifest["selected_frames"], [24])
        self.assertEqual(manifest["actual_backend"], "sam3d")
        self.assertEqual(manifest["fallback_reason"], "only_one_quality_passing_view")

    def test_object_track_identity_is_checked(self):
        (self.root / "object_track.json").write_text(
            json.dumps({"object_id": 8, "prompt": "red pan"}))
        with self.assertRaisesRegex(ValueError, "Object track"):
            self.select({17})

    def test_mismatched_rgb_mask_coordinates_fail(self):
        Image.fromarray(np.ones((12, 32), np.uint8)).save(self.masks / "000017.png")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.select({17})

    def test_reference_mask_must_match_detector_box(self):
        (self.root / "dino_detections.json").write_text(json.dumps([
            {"box": {"x0": 0, "y0": 0, "x1": 3, "y1": 3}}
        ]))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.select({17})

    def test_explicit_single_view_records_source_and_seed(self):
        name = "000017.png"
        manifest = record_single_view(
            output_dir=str(self.root / "mesh"), image_path=str(self.frames / name),
            mask_path=str(self.masks / name), depth_path=str(self.depth / name),
            object_id=7, object_prompt="red pan", seed=12)
        self.assertEqual(manifest["selected_frames"], [17])
        self.assertEqual(manifest["seed"], 12)
        self.assertEqual(manifest["actual_backend"], "sam3d")
        self.assertTrue((self.root / "mesh/mesh_generation_manifest.json").is_file())

    def test_raw_mv_mesh_feeds_obj_loader_and_is_cached(self):
        manifest = self.select({3, 17})
        repo = self.root / "vendor"
        repo.mkdir()
        weights = self.root / "weights/hf-download/checkpoints"
        weights.mkdir(parents=True)
        (weights / "pipeline.yaml").write_text("model: test\n")
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            result = self.root / "mesh_mv/mv_run/test/result.glb"
            result.parent.mkdir(parents=True)
            trimesh.creation.box().export(result)

        with patch("mv_sam3d_mesh.subprocess.check_output", return_value=MV_SAM3D_COMMIT), \
             patch("mv_sam3d_mesh.subprocess.run", side_effect=fake_run):
            obj = run_mv_sam3d(manifest=manifest, repo_dir=str(repo),
                               weights_dir=str(weights), docker_image="test:latest")
            again = run_mv_sam3d(manifest=manifest, repo_dir=str(repo),
                                 weights_dir=str(weights), docker_image="test:latest")
        self.assertEqual(obj, again)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("--image_names") + 1], "0,1")
        self.assertEqual(len(trimesh.load(obj).faces), 12)
        generation = json.loads(Path(manifest["manifest_path"]).read_text())["generation"]
        self.assertEqual(generation["mv_sam3d_commit"], MV_SAM3D_COMMIT)
        self.assertTrue(generation["checkpoint_config_sha256"])


if __name__ == "__main__":
    unittest.main()
