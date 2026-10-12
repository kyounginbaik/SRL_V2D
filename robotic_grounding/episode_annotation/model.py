"""Structured OpenAI Responses calls and semantic analysis pipeline."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .core import Config, Episode, config_dict, interval_frames, sample_frames, save_json, validate_annotations


PROMPT_VERSION = "episode-annotation-v1"
SEGMENT_PROMPT = (
    "Analyze this robot-manipulation episode from chronologically ordered observed frames. "
    "A stage is a contiguous interval with a relatively consistent immediate manipulation goal and hand-object roles. "
    "Granularity: {granularity}. Low merges related sub-actions serving one immediate goal. "
    "Mid names meaningful stages such as approach, grasp, lift, transport, place, release when visible. "
    "High allows meaningful regrasping, alignment and hand-role changes. These examples are not a mandatory sequence. "
    "Do not split for incidental motion, camera movement, or tiny direction changes. Preserve short meaningful actions. "
    "Describe anatomical left and right hand roles separately; image-left is not necessarily anatomical left. "
    "Identify objects consistently, summarize the task, and propose only stages supported by the images. "
    "All text must be English. Candidate starts use original source frame IDs. "
    "For every stage after the first, give observed frames on either side of its proposed transition where identifiable; "
    "use null if unavailable. Never claim to have observed an unsent frame. "
    "Suggest informative observed key frames for each stage. Do not analyze contact labels in this pass."
)
REFINE_PROMPT = (
    "Refine ONE candidate stage boundary using the local chronological images and the coarse stage summary below. "
    "Return an estimated source-frame start of the new stage, plus the last actually observed frame clearly before "
    "the change and first actually observed frame clearly after it, when identifiable. "
    "Evidence frame IDs must be among the sent images. Keep uncertainty when the exact transition is hidden or "
    "between frames. Do not invent an observation. All text must be English. Coarse summary: {summary}"
)
KEYFRAME_PROMPT = (
    "Select 2-3 informative observed source frame IDs for this stage, or one if only one is available. "
    "Prefer frames that show the stage's action and relevant state changes. "
    "IDs must be among the sent images and within the stage interval. Do not invent frames. "
    "All text must be English. Stage: {stage}"
)
CONTACT_PROMPT = (
    "Estimate binary physical contact at the SPECIFIED key frame for every requested anatomical hand-object pair. "
    "1 means any part of that hand physically touches that object; 0 means estimated absence; "
    "null means insufficient evidence, occlusion, or ambiguity. Image overlap alone is insufficient. "
    "Nearby frames are only supporting temporal evidence; label the specified frame, not the whole stage. "
    "Never infer finger-level contact, force, torque, contact point, or grasp quality. "
    "Do not assume contact stays constant during a stage. Use the given object IDs consistently. "
    "Confidence is qualitative high, medium, or low, not a probability. Give short English evidence for each pair. "
    "Specified frame: {frame_id}; pairs: {pairs}; stage: {stage}"
)


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


S = {"type": "string"}
I = {"type": "integer"}
NI = {"type": ["integer", "null"]}
CONF = {"type": "string", "enum": ["high", "medium", "low"]}
PAIR = obj({"hand": {"type": "string", "enum": ["left", "right"]}, "object_id": S})
COARSE_SCHEMA = obj({
    "task_summary": S,
    "objects": arr(obj({"id": S, "description": S})),
    "stages": arr(obj({"name": S, "description": S, "candidate_start_frame": I,
                       "last_observed_before": NI, "first_observed_after": NI,
                       "left_hand_role": S, "right_hand_role": S,
                       "related_object_ids": arr(S), "relevant_pairs": arr(PAIR),
                       "confidence": CONF, "suggested_key_frame_ids": arr(I)})),
})
REFINE_SCHEMA = obj({"estimated_start_frame": I, "last_observed_before": NI,
                     "first_observed_after": NI, "confidence": CONF, "note": S})
KEYFRAME_SCHEMA = obj({"key_frame_ids": arr(I)})
CONTACT_SCHEMA = obj({"assessments": arr(obj({"hand": {"type": "string", "enum": ["left", "right"]},
                                           "object_id": S, "value": {"type": ["integer", "null"]},
                                           "evidence": S, "confidence": CONF}))})
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
NON_RETRYABLE_RATE_CODES = {"insufficient_quota", "organization_spend_limit_exceeded",
                            "project_spend_limit_exceeded", "organization_usage_limit_exceeded"}


def _retry_after_seconds(header: str | None) -> float | None:
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        try:
            target = parsedate_to_datetime(header)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _error_code(exc: urllib.error.HTTPError) -> str | None:
    try:
        payload = json.load(exc)
        code = payload.get("error", {}).get("code")
        return code if isinstance(code, str) else None
    except (ValueError, TypeError, AttributeError, OSError):
        return None


class ResponsesClient:
    """Small REST client; credentials come only from the environment."""

    def __init__(self, cache_dir: Path, model: str):
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for analysis")
        # HTTP libraries can include a rejected header value in a ValueError.
        # Reject pasted commands, newlines, and other malformed values first.
        if any(not 33 <= ord(character) <= 126 for character in key):
            raise RuntimeError("OPENAI_API_KEY must be one key on one line, without spaces or commands")
        self.cache_dir = cache_dir
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def call(self, label: str, prompt: str, frames: list[int], episode: Episode,
             frame_cache: Path, schema: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame_id in sorted(set(frames)):
            image_path = episode.frame(frame_id, frame_cache)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
            content.append({"type": "input_text", "text": f"Source frame ID: {frame_id}; timestamp: {episode.times[frame_id]:.6f} s"})
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
        body = {"model": self.model, "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": label, "schema": schema, "strict": True}},
                "store": False}
        serialized = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
        key = hashlib.sha256(serialized).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{label}_{key}.json"
        if cache_path.exists():
            print(f"Using cached {label} response", file=sys.stderr, flush=True)
            response = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            print(f"Analyzing {label} ({len(set(frames))} frames)", file=sys.stderr, flush=True)
            started = time.monotonic()
            for attempt in range(5):
                try:
                    request = urllib.request.Request("https://api.openai.com/v1/responses", data=serialized,
                                                     headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                                              "Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(request, timeout=600) as stream:
                        response = json.load(stream)
                    break
                except urllib.error.HTTPError as exc:
                    # Never print the response body, headers, or request: any could include credentials.
                    code = _error_code(exc)
                    retryable = exc.code in RETRYABLE_STATUSES
                    if exc.code == 429 and code in NON_RETRYABLE_RATE_CODES:
                        retryable = False
                    if not retryable or attempt == 4:
                        raise RuntimeError(
                            f"OpenAI Responses request failed with HTTP {exc.code}"
                            + (f" after {attempt + 1} attempts" if attempt else "")
                        ) from None
                    server_delay = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
                    delay = (server_delay if server_delay is not None else min(2 ** (attempt + 1), 20))
                    delay += random.uniform(0.0, 0.5)
                    if time.monotonic() - started + delay > 120:
                        raise RuntimeError(
                            f"OpenAI Responses request returned HTTP {exc.code}; retry later"
                        ) from None
                    print(f"OpenAI request returned HTTP {exc.code}; retrying in {delay:.1f}s "
                          f"(attempt {attempt + 2}/5)", file=sys.stderr, flush=True)
                    time.sleep(delay)
                except Exception as exc:
                    # Some HTTP exceptions include the Authorization header in their message.
                    raise RuntimeError(f"OpenAI Responses request failed ({type(exc).__name__}); credential hidden") from None
            save_json(cache_path, response)
        if response.get("status") != "completed":
            raise RuntimeError(f"OpenAI response did not complete: {response.get('status')}")
        output = [part.get("text") for item in response.get("output", []) if item.get("type") == "message"
                  for part in item.get("content", []) if part.get("type") == "output_text"]
        if not output:
            raise RuntimeError("OpenAI response contained no structured text")
        data = json.loads("".join(output))
        self.calls.append({"step": label, "prompt": prompt, "frame_ids": sorted(set(frames)),
                           "response_id": response.get("id"), "actual_model": response.get("model"),
                           "usage": response.get("usage"), "cache_file": str(cache_path)})
        return data


def _validate_coarse(coarse: dict[str, Any], observed: set[int], count: int) -> None:
    stages = coarse["stages"]
    if not stages or len(stages) > count or stages[0]["candidate_start_frame"] != 0:
        raise ValueError("model proposed invalid stage count or first start")
    starts = [x["candidate_start_frame"] for x in stages]
    if any(not 0 <= x < count for x in starts) or any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError("model stage starts are not strictly ordered")
    ids = {x["id"] for x in coarse["objects"]}
    if len(ids) != len(coarse["objects"]):
        raise ValueError("model object IDs are duplicated")
    for stage in stages:
        if any(x not in ids for x in stage["related_object_ids"]):
            raise ValueError("model stage references unknown object")
        estimate = stage["candidate_start_frame"]
        before, after = stage["last_observed_before"], stage["first_observed_after"]
        if before is not None and (before not in observed or before >= estimate):
            stage["last_observed_before"] = None
        if after is not None and (after not in observed or after < estimate):
            stage["first_observed_after"] = None
        for pair in stage["relevant_pairs"]:
            if pair["hand"] not in {"left", "right"} or pair["object_id"] not in stage["related_object_ids"]:
                raise ValueError("model returned invalid relevant pair")
        if len({(x["hand"], x["object_id"]) for x in stage["relevant_pairs"]}) != len(stage["relevant_pairs"]):
            raise ValueError("model returned duplicate relevant pairs")
        if any(x not in observed for x in stage["suggested_key_frame_ids"]):
            raise ValueError("model suggested an unobserved key frame")


def _boundary(proposal: dict[str, Any], fallback: int, observed: set[int], previous: int, following: int,
              coarse: dict[str, Any]) -> dict[str, Any]:
    estimate = proposal["estimated_start_frame"]
    before = proposal["last_observed_before"]
    after = proposal["first_observed_after"]
    valid = previous < estimate < following
    valid &= before is None or before in observed and before < estimate
    valid &= after is None or after in observed and after >= estimate
    valid &= before is None or after is None or before < after
    if valid:
        return proposal
    coarse_before, coarse_after = coarse["last_observed_before"], coarse["first_observed_after"]
    if coarse_before not in observed or coarse_before is not None and coarse_before >= fallback:
        coarse_before = None
    if coarse_after not in observed or coarse_after is not None and coarse_after < fallback:
        coarse_after = None
    return {"estimated_start_frame": fallback, "last_observed_before": coarse_before,
            "first_observed_after": coarse_after, "confidence": "low",
            "note": "Refinement was inconsistent with inspected frames; coarse estimate retained."}


def analyze(episode: Episode, config: Config, output_dir: Path) -> dict[str, Any]:
    config.validate()
    cache_root = output_dir / "cache" / episode.identity / hashlib.sha256(json.dumps(config_dict(config), sort_keys=True).encode()).hexdigest()[:16]
    frame_cache = cache_root / "frames"
    client = ResponsesClient(cache_root / "api", config.model)
    coarse_ids = sample_frames(episode.times, config.coarse_fps)
    observed = set(coarse_ids)
    coarse_prompt = SEGMENT_PROMPT.format(granularity=config.granularity)
    coarse = client.call("coarse", coarse_prompt, coarse_ids, episode, frame_cache, COARSE_SCHEMA)
    _validate_coarse(coarse, observed, episode.frame_count)
    proposals = coarse["stages"]
    starts = [x["candidate_start_frame"] for x in proposals] + [episode.frame_count]
    boundaries = []
    for index in range(1, len(proposals)):
        center = starts[index]
        local = set(interval_frames(episode.times, center, config.context_seconds, config.refinement_fps))
        local.update((center - 1, center))
        local = sorted(local)
        observed.update(local)
        summary = f"Before: {proposals[index - 1]['name']} - {proposals[index - 1]['description']}. After: {proposals[index]['name']} - {proposals[index]['description']}. Candidate start: {center}."
        result = client.call("boundary", REFINE_PROMPT.format(summary=summary), local, episode, frame_cache, REFINE_SCHEMA)
        boundaries.append(_boundary(result, center, observed, starts[index - 1], starts[index + 1], proposals[index]))
    final_starts = [0] + [x["estimated_start_frame"] for x in boundaries] + [episode.frame_count]
    if any(a >= b for a, b in zip(final_starts, final_starts[1:])):
        raise ValueError("refined boundaries produce empty or reversed stages")
    stages = []
    for index, proposal in enumerate(proposals):
        start, end = final_starts[index:index + 2]
        candidates = sorted(x for x in observed if start <= x < end)
        candidates = sorted(set(candidates) | {start, (start + end - 1) // 2, end - 1})
        observed.update(candidates)
        stage_text = f"{proposal['name']}: {proposal['description']}; interval [{start}, {end}); left: {proposal['left_hand_role']}; right: {proposal['right_hand_role']}"
        selected = client.call("keyframes", KEYFRAME_PROMPT.format(stage=stage_text), candidates, episode,
                               frame_cache, KEYFRAME_SCHEMA)["key_frame_ids"]
        selected = list(dict.fromkeys(selected))
        if not 1 <= len(selected) <= min(3, len(candidates)) or any(x not in candidates for x in selected):
            raise ValueError("model chose invalid or unobserved key frames")
        stages.append({"id": f"S{index + 1:02d}", "name": proposal["name"], "description": proposal["description"],
                       "start_frame": start, "end_frame": end, "start_time": episode.times[start],
                       "end_time_exclusive": episode.time_at_boundary(end),
                       "end_time_is_estimated": end == episode.frame_count,
                       "hand_roles": {"left": proposal["left_hand_role"], "right": proposal["right_hand_role"]},
                       "related_object_ids": proposal["related_object_ids"], "relevant_pairs": proposal["relevant_pairs"],
                       "confidence": proposal["confidence"], "key_frame_ids": selected})
    contacts = []
    if config.contact == "on":
        for stage in stages:
            pairs = stage["relevant_pairs"]
            if not pairs:
                continue
            for key in stage["key_frame_ids"]:
                local = sorted(set(interval_frames(episode.times, key, min(config.context_seconds, 0.5), config.refinement_fps))
                               | {max(0, key - 1), key, min(episode.frame_count - 1, key + 1)})
                observed.update(local)
                prompt = CONTACT_PROMPT.format(frame_id=key, pairs=json.dumps(pairs), stage=stage["name"])
                result = client.call("contact", prompt, local, episode, frame_cache, CONTACT_SCHEMA)
                expected = {(p["hand"], p["object_id"]) for p in pairs}
                got = {(p["hand"], p["object_id"]) for p in result["assessments"]}
                if len(got) != len(result["assessments"]) or got != expected:
                    raise ValueError("model contact pairs differ from requested pairs")
                contacts.extend({"frame_id": key, **assessment} for assessment in result["assessments"])
    data = {"schema_version": 1, "episode": {**episode.metadata(), "model_requested": config.model,
            "actual_models": sorted({x["actual_model"] for x in client.calls if x["actual_model"]}),
            "granularity": config.granularity, "contact": config.contact,
            "sampling": {"coarse_fps": config.coarse_fps, "refinement_fps": config.refinement_fps,
                         "context_seconds": config.context_seconds}},
            "task_summary": coarse["task_summary"], "objects": coarse["objects"], "stages": stages,
            "boundaries": boundaries, "observed_frame_ids": sorted(observed), "contacts": contacts,
            "provenance": {"prompt_version": PROMPT_VERSION,
                           "analysis_config": {**config_dict(config), "output_dir": str(output_dir.resolve())},
                           "calls": client.calls}}
    validate_annotations(data)
    save_json(output_dir / "annotations.json", data)
    return data
