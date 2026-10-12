"""Exercise the full pipeline with deterministic model responses, without network access."""

from __future__ import annotations

import tempfile
import unittest
import os
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from episode_annotation.core import Config, Episode, validate_annotations
from episode_annotation.model import COARSE_SCHEMA, ResponsesClient, analyze


class FakeResponsesClient:
    labels: list[str] = []

    def __init__(self, cache_dir: Path, model: str):
        self.calls = []
        self.model = model

    def call(self, label, prompt, frames, episode, frame_cache, schema):
        self.labels.append(label)
        assert frames == sorted(frames)
        for frame_id in frames:
            assert episode.frame(frame_id, frame_cache).exists()
        self.calls.append({"step": label, "prompt": prompt, "frame_ids": frames,
                           "actual_model": self.model, "usage": {"input_tokens": 1}})
        if label == "coarse":
            return {
                "task_summary": "Lift a small box",
                "objects": [{"id": "box", "description": "Small box"}],
                "stages": [
                    {"name": "Approach", "description": "Left hand approaches the box",
                     "candidate_start_frame": 0, "last_observed_before": None, "first_observed_after": None,
                     "left_hand_role": "Approaches", "right_hand_role": "Inactive",
                     "related_object_ids": ["box"], "relevant_pairs": [{"hand": "left", "object_id": "box"}],
                     "confidence": "high", "suggested_key_frame_ids": [0]},
                    {"name": "Lift", "description": "Left hand lifts the box",
                     "candidate_start_frame": 2, "last_observed_before": 1, "first_observed_after": 2,
                     "left_hand_role": "Lifts", "right_hand_role": "Inactive",
                     "related_object_ids": ["box"], "relevant_pairs": [{"hand": "left", "object_id": "box"}],
                     "confidence": "medium", "suggested_key_frame_ids": [2]},
                ],
            }
        if label == "boundary":
            return {"estimated_start_frame": 2, "last_observed_before": 1,
                    "first_observed_after": 2, "confidence": "medium", "note": "Transition lies between frames"}
        if label == "keyframes":
            return {"key_frame_ids": frames[:2]}
        if label == "contact":
            return {"assessments": [{"hand": "left", "object_id": "box", "value": None,
                                     "evidence": "The hand-box edge is hidden", "confidence": "low"}]}
        raise AssertionError(label)


class PipelineTests(unittest.TestCase):
    def test_temporary_503_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (20, 20), "blue").save(frames / "0.png")
            episode = Episode(frames, fps=1)
            temporary = urllib.error.HTTPError("https://api.openai.com/v1/responses", 503, "overloaded",
                                                {"Retry-After": "0"},
                                                io.BytesIO(b'{"error":{"code":"server_is_overloaded"}}'))
            completed = {"status": "completed", "model": "gpt-6.1-sol",
                         "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-example"}):
                client = ResponsesClient(root / "api", "gpt-6.1-sol")
                with (patch("episode_annotation.model.urllib.request.urlopen",
                            side_effect=[temporary, io.BytesIO(json.dumps(completed).encode())]) as send,
                      patch("episode_annotation.model.time.sleep") as sleep,
                      patch("episode_annotation.model.random.uniform", return_value=0)):
                    self.assertEqual(client.call("coarse", "Analyze", [0], episode,
                                                 root / "cache", COARSE_SCHEMA), {})
                self.assertEqual(send.call_count, 2)
                sleep.assert_called_once_with(0.0)

    def test_quota_429_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (20, 20), "blue").save(frames / "0.png")
            episode = Episode(frames, fps=1)
            quota = urllib.error.HTTPError("https://api.openai.com/v1/responses", 429, "quota",
                                            {"Retry-After": "0"},
                                            io.BytesIO(b'{"error":{"code":"insufficient_quota"}}'))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-example"}):
                client = ResponsesClient(root / "api", "gpt-6.1-sol")
                with patch("episode_annotation.model.urllib.request.urlopen", side_effect=quota) as send:
                    with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                        client.call("coarse", "Analyze", [0], episode, root / "cache", COARSE_SCHEMA)
                self.assertEqual(send.call_count, 1)

    def test_malformed_key_is_rejected_without_echo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            secret = "example-secret-that-must-not-appear"
            with patch.dict(os.environ, {"OPENAI_API_KEY": "pasted command\r" + secret}):
                with self.assertRaises(RuntimeError) as caught:
                    ResponsesClient(Path(temp), "gpt-6.1-sol")
            self.assertNotIn(secret, str(caught.exception))

    def test_transport_error_hides_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (20, 20), "blue").save(frames / "0.png")
            episode = Episode(frames, fps=1)
            secret = "sk-example-secret-that-must-not-appear"
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
                client = ResponsesClient(root / "api", "gpt-6.1-sol")
                with patch("episode_annotation.model.urllib.request.urlopen",
                           side_effect=ValueError("Invalid header value " + secret)):
                    with self.assertRaises(RuntimeError) as caught:
                        client.call("coarse", "Analyze", [0], episode, root / "cache", COARSE_SCHEMA)
            self.assertNotIn(secret, str(caught.exception))

    def test_analysis_and_disabled_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / "frames"
            frames.mkdir()
            for index in range(4):
                Image.new("RGB", (64, 40), (index * 40, 50, 80)).save(frames / f"{index:03d}.png")
            episode = Episode(frames, fps=4)
            with patch("episode_annotation.model.ResponsesClient", FakeResponsesClient):
                FakeResponsesClient.labels = []
                on = analyze(episode, Config(contact="on"), root / "on")
                validate_annotations(on)
                self.assertIn("contact", FakeResponsesClient.labels)
                self.assertEqual([(x["start_frame"], x["end_frame"]) for x in on["stages"]], [(0, 2), (2, 4)])
                self.assertEqual(on["boundaries"][0]["last_observed_before"], 1)
                FakeResponsesClient.labels = []
                off = analyze(episode, Config(contact="off"), root / "off")
                validate_annotations(off)
                self.assertNotIn("contact", FakeResponsesClient.labels)
                self.assertEqual(off["contacts"], [])


if __name__ == "__main__":
    unittest.main()
