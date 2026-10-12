"""Episode input, sampling, caching, and annotation validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CONFIDENCES = {"high", "medium", "low"}


@dataclass(frozen=True)
class Config:
    model: str = "gpt-6.1-sol"
    granularity: str = "mid"
    contact: str = "on"
    coarse_fps: float = 2.0
    refinement_fps: float = 8.0
    context_seconds: float = 1.0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.granularity not in {"low", "mid", "high"}:
            raise ValueError("granularity must be low, mid, or high")
        if self.contact not in {"on", "off"}:
            raise ValueError("contact must be on or off")
        if any(not math.isfinite(x) or x <= 0 for x in (self.coarse_fps, self.refinement_fps, self.context_seconds)):
            raise ValueError("sampling rates and context duration must be positive finite numbers")


def _natural_key(path: Path) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def _read_timestamps(path: Path, names: list[str]) -> list[float]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [float(data[name]) for name in names]
        return [float(x) for x in data]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "timestamp" not in rows[0]:
        raise ValueError("timestamp CSV needs a timestamp column")
    if "filename" in rows[0]:
        lookup = {row["filename"]: float(row["timestamp"]) for row in rows}
        return [lookup[name] for name in names]
    return [float(row["timestamp"]) for row in rows]


def _check_timestamps(times: list[float], count: int) -> None:
    if len(times) != count or count == 0:
        raise ValueError("timestamps must match the nonempty frame sequence")
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("timestamps must be finite and nonnegative")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must increase strictly")


class Episode:
    def __init__(self, path: Path, fps: float | None = None, timestamps: Path | list[float] | None = None):
        self.path = path.expanduser().resolve()
        self.fps_override = fps is not None
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.kind = "frames" if self.path.is_dir() else "video"
        self.files: list[Path] = []
        if self.kind == "frames":
            self.files = sorted((x for x in self.path.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
            if not self.files:
                raise ValueError("frame directory contains no RGB images")
            if timestamps is None and fps is None:
                raise ValueError("frame directories require --fps or --timestamps")
            if timestamps is not None:
                self.times = list(timestamps) if isinstance(timestamps, list) else _read_timestamps(timestamps, [x.name for x in self.files])
                self.fps = None
            else:
                if fps is None or not math.isfinite(fps) or fps <= 0:
                    raise ValueError("fps must be positive")
                self.fps = fps
                self.times = [i / fps for i in range(len(self.files))]
        else:
            if timestamps is not None:
                raise ValueError("--timestamps applies to RGB frame directories")
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Video input requires opencv-python") from exc
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise ValueError(f"cannot open video: {self.path}")
            self.fps = float(fps if fps is not None else cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.fps) or self.fps <= 0:
                cap.release()
                raise ValueError("video FPS is unavailable; provide --fps")
            self.times = []
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                index = len(self.times)
                observed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                expected = index / self.fps
                use_decoder_time = fps is None and (index == 0 or observed > self.times[-1])
                self.times.append(observed if use_decoder_time else expected)
            cap.release()
            # Some codecs report a constant or imprecise POS_MSEC. FPS is the safe fallback.
            if any(b <= a for a, b in zip(self.times, self.times[1:])):
                self.times = [i / self.fps for i in range(len(self.times))]
        _check_timestamps(self.times, len(self.files) if self.kind == "frames" else len(self.times))
        self.frame_count = len(self.times)
        self.identity = self._identity()

    def _identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.path).encode())
        sources = self.files if self.kind == "frames" else [self.path]
        for source in sources:
            digest.update(source.name.encode())
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(json.dumps(self.times).encode())
        return digest.hexdigest()

    def frame(self, frame_id: int, cache_dir: Path) -> Path:
        if not 0 <= frame_id < self.frame_count:
            raise ValueError(f"frame {frame_id} is outside the episode")
        from PIL import Image

        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"frame_{frame_id:08d}.png"
        if target.exists():
            return target
        if self.kind == "frames":
            with Image.open(self.files[frame_id]) as image:
                rgb = image.convert("RGB")
                rgb.save(target)
        else:
            import cv2

            cap = cv2.VideoCapture(str(self.path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, data = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"video frame {frame_id} could not be decoded")
            Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB)).save(target)
        return target

    def metadata(self) -> dict[str, Any]:
        return {"input_path": str(self.path), "input_identity": self.identity, "input_type": self.kind,
                "frame_count": self.frame_count, "fps": self.fps, "fps_override": self.fps_override,
                "timestamps": self.times,
                "frame_files": [x.name for x in self.files] if self.kind == "frames" else None}

    def time_at_boundary(self, frame_id: int) -> float:
        if frame_id < self.frame_count:
            return self.times[frame_id]
        if frame_id != self.frame_count:
            raise ValueError("boundary is outside episode")
        if self.fps:
            return self.times[-1] + 1 / self.fps
        if self.frame_count > 1:
            return self.times[-1] + (self.times[-1] - self.times[-2])
        return self.times[-1]


def sample_frames(times: list[float], rate: float, start: int = 0, end: int | None = None) -> list[int]:
    """Sample original IDs by elapsed time, including both endpoints."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive")
    end = len(times) if end is None else end
    if not 0 <= start < end <= len(times):
        raise ValueError("invalid sampling interval")
    selected = [start]
    target = times[start] + 1 / rate
    for index in range(start + 1, end - 1):
        if times[index] + 1e-9 >= target:
            selected.append(index)
            target = times[index] + 1 / rate
    if end - 1 != selected[-1]:
        selected.append(end - 1)
    return selected


def interval_frames(times: list[float], center: int, seconds: float, rate: float) -> list[int]:
    lower = next((i for i, t in enumerate(times) if t >= times[center] - seconds), 0)
    upper = next((i for i, t in enumerate(times) if t > times[center] + seconds), len(times))
    return sample_frames(times, rate, lower, upper)


def _valid_confidence(value: Any) -> bool:
    return value in CONFIDENCES


def _require(mapping: Any, fields: dict[str, type], location: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{location} must be an object")
    for name, expected in fields.items():
        if name not in mapping or not isinstance(mapping[name], expected):
            raise ValueError(f"{location}.{name} must be {expected.__name__}")


def validate_annotations(data: dict[str, Any]) -> None:
    """Validate persisted data before visualization or reuse."""
    _require(data, {"schema_version": int, "episode": dict, "objects": list,
                    "stages": list, "boundaries": list, "observed_frame_ids": list,
                    "contacts": list, "provenance": dict}, "annotations")
    if data["schema_version"] != 1:
        raise ValueError("unsupported annotation schema version")
    metadata = data["episode"]
    _require(metadata, {"input_path": str, "input_identity": str, "input_type": str,
                        "frame_count": int, "fps_override": bool, "timestamps": list, "model_requested": str,
                        "granularity": str, "contact": str, "sampling": dict,
                        "actual_models": list}, "episode")
    count = metadata["frame_count"]
    times = metadata["timestamps"]
    _check_timestamps(times, count)
    if metadata["contact"] not in {"on", "off"}:
        raise ValueError("invalid contact setting")
    objects = data["objects"]
    for obj in objects:
        _require(obj, {"id": str, "description": str}, "object")
    object_ids = [obj["id"] for obj in objects]
    if len(set(object_ids)) != len(object_ids) or any(not x for x in object_ids):
        raise ValueError("object IDs must be unique and nonempty")
    stages = data["stages"]
    if not stages or stages[0]["start_frame"] != 0 or stages[-1]["end_frame"] != count:
        raise ValueError("stages must cover the full episode")
    seen_stage_ids = set()
    all_keyframes = set()
    for i, stage in enumerate(stages):
        _require(stage, {"id": str, "name": str, "description": str, "start_frame": int,
                         "end_frame": int, "hand_roles": dict, "related_object_ids": list,
                         "relevant_pairs": list, "confidence": str, "key_frame_ids": list}, "stage")
        start, end = stage["start_frame"], stage["end_frame"]
        if stage["id"] in seen_stage_ids or not 0 <= start < end <= count:
            raise ValueError("duplicate stage ID or invalid interval")
        seen_stage_ids.add(stage["id"])
        if i and stages[i - 1]["end_frame"] != start:
            raise ValueError("stage intervals have a gap or overlap")
        end_time = times[end] if end < count else times[-1] + (1 / metadata["fps"] if metadata["fps"] else
                                                                   (times[-1] - times[-2] if count > 1 else 0))
        if (not isinstance(stage.get("start_time"), (int, float)) or
                not isinstance(stage.get("end_time_exclusive"), (int, float)) or
                not math.isclose(stage["start_time"], times[start], abs_tol=1e-6) or
                not math.isclose(stage["end_time_exclusive"], end_time, abs_tol=1e-6) or
                stage.get("end_time_is_estimated") is not (end == count)):
            raise ValueError("stage time range does not match source-frame timestamps")
        if not _valid_confidence(stage["confidence"]):
            raise ValueError("invalid stage confidence")
        if any(x not in object_ids for x in stage["related_object_ids"]):
            raise ValueError("stage references unknown object")
        keys = stage["key_frame_ids"]
        if (not 1 <= len(keys) <= 3 or any(not isinstance(k, int) for k in keys) or
                len(set(keys)) != len(keys) or any(not start <= k < end for k in keys)):
            raise ValueError("stage key frames must be observed IDs inside the stage")
        all_keyframes.update(keys)
        for pair in stage["relevant_pairs"]:
            _require(pair, {"hand": str, "object_id": str}, "relevant pair")
            if pair["hand"] not in {"left", "right"} or pair["object_id"] not in stage["related_object_ids"]:
                raise ValueError("invalid relevant hand-object pair")
        if len({(x["hand"], x["object_id"]) for x in stage["relevant_pairs"]}) != len(stage["relevant_pairs"]):
            raise ValueError("duplicate relevant hand-object pair")
    boundaries = data["boundaries"]
    if len(boundaries) != len(stages) - 1:
        raise ValueError("one boundary record is required between each stage")
    observed = set(data["observed_frame_ids"])
    if (any(not isinstance(x, int) or not 0 <= x < count for x in observed) or
            len(observed) != len(data["observed_frame_ids"])):
        raise ValueError("invalid observed frame IDs")
    if not all_keyframes <= observed:
        raise ValueError("key frames must have been inspected by the model")
    for i, boundary in enumerate(boundaries):
        _require(boundary, {"estimated_start_frame": int, "confidence": str, "note": str}, "boundary")
        if "last_observed_before" not in boundary or "first_observed_after" not in boundary:
            raise ValueError("boundary evidence fields are required")
        at = stages[i + 1]["start_frame"]
        if boundary["estimated_start_frame"] != at:
            raise ValueError("boundary estimate disagrees with stage interval")
        before, after = boundary["last_observed_before"], boundary["first_observed_after"]
        if before is not None and (before not in observed or before >= at):
            raise ValueError("invalid observed before-boundary frame")
        if after is not None and (after not in observed or after < at):
            raise ValueError("invalid observed after-boundary frame")
        if before is not None and after is not None and before >= after:
            raise ValueError("boundary evidence frames are reversed")
        if not _valid_confidence(boundary["confidence"]):
            raise ValueError("invalid boundary confidence")
    contacts = data["contacts"]
    if metadata["contact"] == "off" and contacts:
        raise ValueError("contact-disabled annotations must contain no assessments")
    expected = {(k, p["hand"], p["object_id"]) for s in stages for k in s["key_frame_ids"] for p in s["relevant_pairs"]}
    actual = set()
    for contact in contacts:
        _require(contact, {"frame_id": int, "hand": str, "object_id": str,
                           "evidence": str, "confidence": str}, "contact")
        if "value" not in contact:
            raise ValueError("contact value is required")
        key = (contact["frame_id"], contact["hand"], contact["object_id"])
        if key in actual or contact["value"] not in (0, 1, None) or isinstance(contact["value"], bool):
            raise ValueError("duplicate contact or invalid contact value")
        if not _valid_confidence(contact["confidence"]) or not contact["evidence"].strip():
            raise ValueError("contact needs evidence and qualitative confidence")
        actual.add(key)
    if metadata["contact"] == "on" and actual != expected:
        raise ValueError("contact assessments do not match relevant key-frame pairs")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def config_dict(config: Config) -> dict[str, Any]:
    return asdict(config)
