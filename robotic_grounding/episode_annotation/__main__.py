"""Command line interface for independent episode annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import Config, Episode
from .model import analyze
from .render import render


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate manipulation stages and optional visual hand-object contact.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("analyze", help="Call the model and produce annotations plus review images")
    run.add_argument("input", type=Path, help="Video file or RGB frame directory")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--fps", type=float, help="Required for frame directories unless --timestamps is supplied")
    run.add_argument("--timestamps", type=Path, help="JSON list/map or CSV timestamp metadata for frame directories")
    run.add_argument("--model", default="gpt-6.1-sol")
    run.add_argument("--granularity", choices=["low", "mid", "high"], default="mid")
    run.add_argument("--contact", choices=["on", "off"], default="on")
    run.add_argument("--coarse-fps", type=float, default=2.0)
    run.add_argument("--refinement-fps", type=float, default=8.0)
    run.add_argument("--context-seconds", type=float, default=1.0)
    rerender = sub.add_parser("render", help="Recreate visualizations from saved annotations, without API calls")
    rerender.add_argument("annotations", type=Path)
    rerender.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        outputs = render(args.annotations, args.output_dir)
    else:
        config = Config(args.model, args.granularity, args.contact, args.coarse_fps,
                        args.refinement_fps, args.context_seconds)
        config.validate()
        episode = Episode(args.input, fps=args.fps, timestamps=args.timestamps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        analyze(episode, config, args.output_dir)
        outputs = render(args.output_dir / "annotations.json")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
