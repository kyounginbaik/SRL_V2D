#!/usr/bin/env python3
"""Package complete validated real rollouts with the official participant kit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from contract import atomic_json, digest, load_roster, validate_kit_roster, validate_trajectory


def inspect_results(results, *, fallback=None, roster=None):
    roster = load_roster() if roster is None else roster
    selected, invalid, missing = {}, {}, []
    for key in sorted(roster, key=int):
        filename = f"episode_{int(key):06d}.parquet"
        options = [Path(results) / filename]
        if fallback is not None:
            options.append(Path(fallback) / filename)
        errors = []
        for path in options:
            if not path.is_file():
                continue
            try:
                report = validate_trajectory(path, episode=key, roster=roster)
            except Exception as error:
                errors.append(dict(path=str(path), error=str(error)))
                continue
            selected[key] = dict(path=str(path.resolve()), fallback=(path != options[0]), **report)
            break
        if errors:
            invalid[key] = errors
        if key not in selected:
            missing.append(int(key))
    return dict(ready=False, expected_episodes=len(roster), available_episodes=len(selected),
                missing_episodes=missing, invalid_inputs=invalid, selected=selected)


def pack(results, output, *, kit, code_commit, fallback=None, allow_incomplete=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".readiness.json")
    report = inspect_results(results, fallback=fallback)
    report.update(code_commit=code_commit, output=str(output),
                  readiness_scope="Submission schema and provenance; task success is measured separately.")
    atomic_json(report_path, report)
    if report["missing_episodes"]:
        print(f"Incomplete: missing valid rollouts for {report['missing_episodes']}", flush=True)
        return 0 if allow_incomplete else 2
    try:
        if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}", code_commit):
            raise ValueError("--code-commit must pin a published GitHub /commit/<40-character SHA> URL")
        if kit is None:
            raise ValueError("Set --kit or V2D_SUBMISSION_KIT to the official participant kit")
        kit = Path(kit).resolve()
        validate_kit_roster(kit)
        with tempfile.TemporaryDirectory(prefix=".track3-pack-", dir=output.parent) as temporary:
            staged = Path(temporary) / "rollouts"
            staged.mkdir()
            for key, item in report["selected"].items():
                target = staged / f"episode_{int(key):06d}.parquet"
                shutil.copy2(item["path"], target)
                if digest(target) != item["trajectory_sha256"]:
                    raise ValueError(f"Episode {key} changed while staging the submission")
            candidate = Path(temporary) / "submission.csv"
            command = [sys.executable, str(kit / "tools/pack_submission.py"),
                       "--track", "track_3", "--results", str(staged),
                       "--code-commit", code_commit, "-o", str(candidate)]
            subprocess.run(command, check=True)
            if not candidate.is_file() or not candidate.stat().st_size:
                raise ValueError("Official packer produced no submission CSV")
            os.replace(candidate, output)
        report.update(ready=True, submission_sha256=digest(output),
                      official_packer_sha256=digest(kit / "tools/pack_submission.py"))
        atomic_json(report_path, report)
        print(f"Ready: {output} ({report['available_episodes']} episodes)", flush=True)
        return 0
    except Exception as error:
        report["error"] = str(error)
        atomic_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--fallback-results", type=Path,
                        help="Optional first-pass rollouts used only when a full result is absent/invalid")
    parser.add_argument("--kit", type=Path, default=os.environ.get("V2D_SUBMISSION_KIT"))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Write an explicit not-ready report and exit successfully while jobs remain")
    args = parser.parse_args()
    raise SystemExit(pack(args.results, args.output, kit=args.kit, code_commit=args.code_commit,
                          fallback=args.fallback_results, allow_incomplete=args.allow_incomplete))


if __name__ == "__main__":
    main()
