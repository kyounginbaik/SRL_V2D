"""Generate review images from persisted annotations and source images."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .core import Episode, validate_annotations


COLORS = [(44, 103, 191), (33, 145, 112), (172, 91, 44), (120, 74, 164), (193, 72, 93)]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow can locate its bundled DejaVu font even when the distro font path
    # does not exist (for example, on minimal compute nodes).
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    result: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > width:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result or [""]


def _contact_text(data: dict[str, Any], frame_id: int, stage: dict[str, Any]) -> str:
    if data["episode"]["contact"] == "off":
        return "Contact: disabled."
    labels = {(x["hand"], x["object_id"]): x for x in data["contacts"] if x["frame_id"] == frame_id}
    if not stage["relevant_pairs"]:
        return "Contact: no relevant pairs"
    parts = []
    for pair in stage["relevant_pairs"]:
        hand, obj = pair["hand"], pair["object_id"]
        value = labels[(hand, obj)]["value"]
        parts.append(f"{hand} / {obj}: {'?' if value is None else value}")
    return "Contact: " + "; ".join(parts)


def _panel(data: dict[str, Any], episode: Episode, stage: dict[str, Any], frame_id: int,
           cache: Path, stage_index: int) -> Image.Image:
    width, height = 620, 520
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    color = COLORS[stage_index % len(COLORS)]
    draw.rectangle((0, 0, width, 8), fill=color)
    with Image.open(episode.frame(frame_id, cache)) as source:
        image = source.convert("RGB")
        image.thumbnail((width - 24, 330), Image.Resampling.LANCZOS)
        x = (width - image.width) // 2
        y = 18 + (330 - image.height) // 2
        panel.paste(image, (x, y))
    title_font, body_font = _font(26), _font(21)
    y = 358
    for line in _lines(draw, f"{stage['id']} · {stage['name']}", title_font, width - 24)[:2]:
        draw.text((12, y), line, fill=(25, 30, 40), font=title_font)
        y += 31
    draw.text((12, y), f"Source frame {frame_id} · {data['episode']['timestamps'][frame_id]:.3f} s",
              fill=(52, 61, 76), font=body_font)
    y += 27
    for line in _lines(draw, _contact_text(data, frame_id, stage), body_font, width - 24)[:2]:
        draw.text((12, y), line, fill=(30, 38, 52), font=body_font)
        y += 26
    return panel


def render(annotations_path: Path, output_dir: Path | None = None) -> list[Path]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    validate_annotations(data)
    output_dir = (output_dir or annotations_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = data["episode"]
    timestamps = metadata["timestamps"] if metadata["input_type"] == "frames" and metadata["fps"] is None else None
    input_fps = metadata["fps"] if metadata["input_type"] == "frames" or metadata["fps_override"] else None
    episode = Episode(Path(metadata["input_path"]), fps=input_fps, timestamps=timestamps)
    if episode.identity != metadata["input_identity"] or episode.frame_count != metadata["frame_count"]:
        raise ValueError("source episode has changed since annotation")
    if metadata["input_type"] == "frames" and [x.name for x in episode.files] != metadata["frame_files"]:
        raise ValueError("source frame sequence has changed")
    frame_cache = output_dir / "render_frames" / episode.identity
    made: list[Path] = []
    panels: list[tuple[int, Image.Image]] = []
    key_dir = output_dir / "key_frames"
    key_dir.mkdir(exist_ok=True)
    for index, stage in enumerate(data["stages"]):
        for frame_id in stage["key_frame_ids"]:
            panel = _panel(data, episode, stage, frame_id, frame_cache, index)
            target = key_dir / f"{stage['id']}_frame_{frame_id:08d}.png"
            panel.save(target)
            made.append(target)
            panels.append((index, panel))
    # Up to 12 panels per page. A stage heading remains implicit in each panel.
    for page_index in range(math.ceil(len(panels) / 12)):
        page = panels[page_index * 12:(page_index + 1) * 12]
        rows = math.ceil(len(page) / 2)
        row_height = panels[0][1].height + 16
        sheet = Image.new("RGB", (1264, rows * row_height + 58), (243, 246, 250))
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 10), f"Episode stages · page {page_index + 1}", font=_font(30), fill=(20, 29, 42))
        for position, (_, panel) in enumerate(page):
            sheet.paste(panel, (12 + (position % 2) * 632, 54 + (position // 2) * row_height))
        name = "stage_contact_sheet.png" if page_index == 0 else f"stage_contact_sheet_{page_index + 1:02d}.png"
        target = output_dir / name
        sheet.save(target)
        made.append(target)
    timeline = Image.new("RGB", (1500, max(240, 100 + len(data["stages"]) * 72)), "white")
    draw = ImageDraw.Draw(timeline)
    draw.text((25, 14), "Episode stage timeline", font=_font(31), fill=(22, 31, 44))
    left, right = 360, 1460
    count = metadata["frame_count"]
    label_font, range_font = _font(21), _font(16)
    for index, stage in enumerate(data["stages"]):
        y = 72 + index * 72
        color = COLORS[index % len(COLORS)]
        draw.text((25, y + 6), stage["id"], font=label_font, fill=(25, 35, 50))
        for line_index, line in enumerate(_lines(draw, stage["name"], label_font, left - 96)[:2]):
            draw.text((84, y + 4 + line_index * 25), line, font=label_font, fill=(25, 35, 50))
        x0 = left + (right - left) * stage["start_frame"] / count
        x1 = left + (right - left) * stage["end_frame"] / count
        draw.rounded_rectangle((x0, y, x1, y + 35), radius=5, fill=color)
        range_text = f"[{stage['start_frame']}, {stage['end_frame']})"
        if draw.textlength(range_text, font=range_font) + 10 < x1 - x0:
            draw.text((x0 + 5, y + 7), range_text, font=range_font, fill="white")
        for frame_id in stage["key_frame_ids"]:
            x = left + (right - left) * frame_id / count
            draw.ellipse((x - 4, y + 39, x + 4, y + 47), fill=color)
    draw.line((left, 62, left, timeline.height - 22), fill=(120, 128, 140), width=2)
    draw.line((right, 62, right, timeline.height - 22), fill=(120, 128, 140), width=2)
    draw.text((left, timeline.height - 25), "frame 0", font=_font(16), fill=(60, 70, 83))
    draw.text((right - 95, timeline.height - 25), f"frame {count}", font=_font(16), fill=(60, 70, 83))
    target = output_dir / "stage_timeline.png"
    timeline.save(target)
    made.append(target)
    return made
