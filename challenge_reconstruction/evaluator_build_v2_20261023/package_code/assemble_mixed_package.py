#!/usr/bin/env python3
"""Assemble thirteen freshly evaluated components without changing old releases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from common_evaluator import (official_validate, read, sha, validate_meshes,
                              verify_build, write)


def run(args):
    build = verify_build(args.build_manifest)
    old = read(args.original_package / "manifest.json")
    episodes = old["available_episodes"]
    if episodes != [1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38]:
        raise ValueError("Unexpected original thirteen-episode scope")
    if args.output.exists() or (args.archive and Path(str(args.output) + ".tar").exists()):
        raise FileExistsError("Preserve existing release: " + str(args.output))
    old_inventory = read(args.original_package / "SHA256SUMS.json")
    for relative, expected in old_inventory.items():
        if sha(args.original_package / relative) != expected:
            raise ValueError("Original release changed: " + relative)
    if not args.episode37_archive.is_file():
        raise ValueError("Episode 37 needs its own reproducibility archive")
    selected = []
    for episode in episodes:
        name = f"episode_{episode:06d}"
        directory = args.components_root / name
        report = read(directory / "completed.json")
        if (report["episode"] != name or not report["fresh_backend_executed"] or
                report["eval_code_sha256"] != build["eval_code_sha256"] or
                report["eval_script_sha256"] != build["eval_script_sha256"] or
                report["build_manifest_sha256"] != sha(args.build_manifest) or
                report["code_commit_url"] != build["code_commit_url"]):
            raise ValueError("Component is not a fresh execution under this common build: " + name)
        if episode != 37 and not report.get("exact_previous_physical_states_actions"):
            raise ValueError("Legacy component did not reproduce its reviewed physics: " + name)
        if episode == 37:
            comparison = report.get("metrics", {}).get("original_demo_comparison", {})
            if args.episode37_video_evidence:
                binding = read(args.episode37_video_evidence)
                if (binding.get("video_sha256") != sha(args.episode37_video) or
                        binding.get("actions_sha256") != report["actions"]["sha256"] or
                        binding.get("rendered_from_fresh_recording") is not True):
                    raise ValueError("Fresh episode 37 video does not bind to this actual recording")
            elif comparison.get("numerically_matches_original_demo") is not True:
                raise ValueError("Fresh episode 37 differs from supplied epi37.mp4; render the fresh recording and provide --episode37-video-evidence")
        for key in ("checkpoint", "actions", "backend_export", "trajectory"):
            if sha(report[key]["path"]) != report[key]["sha256"]:
                raise ValueError("Completed evaluation artifact changed: " + name + "/" + key)
        for mesh in report["meshes"].values():
            if sha(mesh["path"]) != mesh["sha256"]:
                raise ValueError("Completed evaluation mesh changed: " + name)
        validation = official_validate(Path(report["trajectory"]["path"]), args.kit,
                                       episode, report["checkpoint"]["sha256"])
        validate_meshes(directory / "meshes", args.kit, episode, validation["object_names"])
        selected.append((name, directory, report, validation))

    # Validate everything before making a new release directory.
    args.output.mkdir(parents=True)
    rows = []
    for name, directory, report, validation in selected:
        episode = int(name.split("_")[-1])
        for folder in ("results", "meshes"):
            shutil.copytree(directory / folder, args.output / folder, dirs_exist_ok=True)
        evidence = args.output / "evidence" / name
        evidence.mkdir(parents=True)
        shutil.copy2(directory / "completed.json", evidence / "common_execution.json")
        shutil.copy2(directory / "build_manifest.json", evidence / "build_manifest.json")
        raw_name = Path(report["backend_export"]["path"]).name
        shutil.copy2(report["backend_export"]["path"], evidence / ("backend_" + raw_name))
        actions = args.output / "robot_actions" / (name + ".npz")
        actions.parent.mkdir(exist_ok=True)
        shutil.copy2(report["actions"]["path"], actions)
        portable = args.output / "portable_archives" / (name + "_portable.tar.gz")
        portable.parent.mkdir(exist_ok=True)
        if episode == 37:
            shutil.copy2(args.episode37_archive, portable)
            if args.episode37_evidence:
                shutil.copytree(args.episode37_evidence, evidence / "backend_review")
            video = args.output / "videos" / name / "epi37.mp4"
            video.parent.mkdir(parents=True)
            shutil.copy2(args.episode37_video, video)
            if args.episode37_video_evidence:
                shutil.copy2(args.episode37_video_evidence, evidence / "video_binding.json")
        else:
            shutil.copy2(args.original_package / "portable_archives" / portable.name, portable)
            shutil.copytree(args.original_package / "evidence" / name, evidence / "original_review")
            shutil.copytree(args.original_package / "videos" / name, args.output / "videos" / name)
        rows.append(dict(episode=name, backend=report["backend_kind"], expected_rows=validation["rows"],
            checkpoint_sha256=report["checkpoint"]["sha256"],
            trajectory_sha256=report["trajectory"]["sha256"],
            common_execution=str(evidence.relative_to(args.output) / "common_execution.json"),
            actions=str(actions.relative_to(args.output)), actions_sha256=sha(actions),
            portable_archive=str(portable.relative_to(args.output)), archive_sha256=sha(portable),
            metrics=report["metrics"],
            previous_physical_recording_exact=report.get("exact_previous_physical_states_actions")))
    code_root = args.output / "code_build"
    for record in build["sources"]:
        destination = code_root / "sources" / record["key"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["path"], destination)
    shutil.copy2(args.build_manifest, code_root / "build_manifest.json")
    manifest = dict(schema="mixed_backend_thirteen_episode_components_v1",
        scope="The thirteen requested episodes; episode 37 replaced with the user-selected ilchold run",
        original_package=str(args.original_package.resolve()), original_manifest_sha256=sha(args.original_package / "manifest.json"),
        original_package_preserved=True, available_episodes=episodes, requested_episode_count=len(episodes),
        available_episode_components_ready=True, fresh_common_evaluator_execution=True,
        evaluator={key: build[key] for key in ("eval_script_sha256", "eval_code_sha256", "code_commit_url")},
        build_manifest_sha256=sha(args.build_manifest), episodes=rows,
        official_required_episodes=old["official_required_episodes"],
        missing_official_source_episodes=old["missing_official_source_episodes"],
        official_roster_unchanged=True, complete_official_submission_ready=False,
        source_pose_proxy_not_official_ground_truth=True, official_runtime_certification=False, uploaded=False)
    write(args.output / "manifest.json", manifest)
    (args.output / "README.md").write_text("""# Thirteen episode components, with ilchold episode 37

Episode 37 uses the policy2508800/reference0005 physical run shown in epi37.mp4.
The other twelve controllers and source reconstructions are unchanged. Each was
executed freshly through the same versioned dispatcher. Their complete physical
state/action sequences exactly match the earlier reviewed recordings.

`results/` contains freshly exported achieved object poses. `meshes/` contains
the corresponding body-frame visual geometry. `robot_actions/` stores the actual
backend recording, with its native action/state convention retained in the
per-episode evidence. The two backends have different rates and schemas: do not
interpret FlashSAC raw policy actions as MuJoCo position targets.

`evidence/` preserves original backend attribution and the new common execution
reports. `code_build/` includes the complete recorded evaluator source manifest
and its source bytes. The shared evaluator really launched each backend; these
are not old rollout files with substituted provenance. `portable_archives/`
retains the original reproducible legacy bundles for twelve episodes (therefore
their original evaluator hashes) and a separate ilchold bundle for episode 37.
The common execution report binds each preserved bundle to the fresh export.

Extract each episode archive into a separate directory and follow its README.
Legacy bundles use CPU MuJoCo; episode 37 uses its documented GPU runtime. The
portable archives intentionally preserve the original underlying lineage.

The reconstructed object references are evaluation proxies, not hidden ground
truth. Passing their MP-SR checks does not promise an official leaderboard score.
This is still a thirteen-episode component package. The official packer requires
all twenty scored episodes; the other seven remain outside the requested scope.
No upload was performed. Original package and components remain untouched.
""")
    inventory = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write(args.output / "SHA256SUMS.json", inventory)
    if args.archive:
        archive = Path(str(args.output) + ".tar")
        with tarfile.open(archive, "w") as packed:
            packed.add(args.output, arcname=args.output.name)
        write(Path(str(archive) + ".json"), dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)))
    print(json.dumps(dict(output=str(args.output), episodes=len(rows), uploaded=False)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("build-manifest", "original-package", "components-root", "kit", "output", "episode37-archive", "episode37-video"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--episode37-evidence", type=Path)
    p.add_argument("--episode37-video-evidence", type=Path,
                   help="For a fresh rerender: JSON binding video_sha256, actions_sha256, rendered_from_fresh_recording=true")
    p.add_argument("--archive", action="store_true")
    run(p.parse_args())
