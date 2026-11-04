#!/usr/bin/env python3
"""Offline, fail-closed Track 3 artifact audit. Never generates poses or provenance.

Five-episode research scope is intentionally distinct from the 20-episode upload roster.
Historical policies are inventory only; no candidate is selected automatically.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

B = Path(__file__).resolve().parents[1]
REPO = B.parents[1]
DEFAULT_OUT = B / "refinement_20261017/submission_readiness"
DEFAULT_KIT = Path("/data1/video_to_data_challenge/v2d_submission_kit")
DEFAULT_CURRENT_KIT = DEFAULT_OUT / "latest_official/v2d_submission_kit"
EPISODES = [1, 3, 5, 8, 10]
ALGORITHMS = {"flash_chord.training.flash_sac.config.TrainingConfig": "flash_sac",
              "flash_chord.training.ppo.config.TrainingConfig": "ppo"}
FAQ_URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf"
KIT_URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def kit_info(kit):
    cfg = read(kit / "config/tracks.json")
    spec = cfg["tracks"]["track_3"]
    manifest = read(kit / "KIT_MANIFEST.json")
    mismatches = []
    for name, expected in manifest.get("files", {}).items():
        path = kit / name
        if not path.is_file() or digest(path) != expected:
            mismatches.append(name)
    return dict(path=str(kit), manifest_sha256=digest(kit / "KIT_MANIFEST.json"),
                manifest_file_mismatches=mismatches, runtime=manifest.get("evaluator"),
                required_episodes=spec["scored_episodes"], worlds_per_episode=spec.get("worlds_per_episode", 1),
                format="combined_parquet_with_meshes" if spec.get("register_object_mesh") else "legacy_pose_csv_separate_mesh_parquet",
                symmetry_aware=bool(spec.get("register_object_mesh")),
                accepted_eval_script_sha256=cfg.get("accepted_eval_script_sha256", []),
                accepted_eval_code_sha256=cfg.get("accepted_eval_code_sha256", []))


def public_roster(kit):
    roster = read(kit / "data/track_3_evaluation_objects.json")["episodes"]
    grid = {}
    sample = kit / "data/track_3_sample_submission.parquet"
    if sample.exists():
        row_ids = pq.read_table(sample, columns=["row_id"])["row_id"].to_pylist()
    else:
        with (kit / "data/track_3_sample_submission.csv").open() as source:
            row_ids = [row["row_id"] for row in csv.DictReader(source)]
    for row_id in row_ids:
        match = re.fullmatch(r"track_3/e(\d+)/w(\d+)/f(\d+)/o(\d+)", row_id)
        if not match:
            raise ValueError("Unrecognized official row ID")
        ep, world, frame, obj = map(int, match.groups())
        grid.setdefault(ep, []).append((world, frame, obj))
    return {ep: dict(object_names=roster[str(ep)], rows=sorted(rows),
                     frames=sorted({r[1] for r in rows}), worlds=sorted({r[0] for r in rows}))
            for ep, rows in grid.items()}


def checkpoint_info(path):
    """Inspect safetensors JSON header only; never unpickle or import saved executable code."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("This audit supports actual FlashSAC/PPO safetensors checkpoints only")
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        count = struct.unpack("<Q", raw)[0]
        if count > 64 * 1024 * 1024 or count + 8 > path.stat().st_size:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(count))
    meta = header.get("__metadata__", {})
    cfg = json.loads(meta.get("user.resolved_config_json", "{}"))
    schema = json.loads(meta.get("user.policy_schema_json", "{}"))
    target = cfg.get("training", {}).get("_target_")
    task = cfg.get("task", {})
    reference = Path(task.get("parquet", ""))
    refined = task.get("robot_reference_path")
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                algorithm=meta.get("user.algorithm"), training_target=target,
                supported_official_algorithm=target in ALGORITHMS,
                checkpoint_kind=meta.get("user.checkpoint_kind"),
                environment_steps=int(meta.get("user.environment_steps", 0)),
                action_dimension=schema.get("action", {}).get("dimension"),
                observation_dimension=schema.get("observation", {}).get("dimension"),
                task=task, embodiment=cfg.get("embodiment"), collision=cfg.get("collision"),
                action=cfg.get("env", {}).get("action"), sim=cfg.get("env", {}).get("sim"), scene=cfg.get("scene"),
                reference_exists=reference.is_file(), reference_sha256=digest(reference) if reference.is_file() else None,
                refined_reference_exists=Path(refined).is_file() if refined else None,
                compatibility_note="Saved robot_reference_path requires runtime support; pinned official aee95f25 harness does not forward it" if refined else None,
                retraining_or_replay_verified=False)


def trajectory_info(path, expected, kit, checkpoint=None):
    # The official reader checks actual schema, pose finiteness and conventions.
    sys.path.insert(0, str(kit))
    from v2dlb.submission_parquet import read_trajectory, provenance
    trajectory = read_trajectory(Path(path))
    table = pq.read_table(path)
    problems = []
    actual = list(zip(trajectory.world_index.tolist(), trajectory.frame_index.tolist(), trajectory.object_slot.tolist()))
    if len(set(actual)) != len(actual):
        problems.append("duplicate world/frame/object rows")
    if sorted(actual) != expected["rows"]:
        problems.append("rows do not exactly match full official episode roster")
    names = trajectory.object_name
    if names is None or any(str(n) != expected["object_names"][int(s)] for n, s in zip(names, trajectory.object_slot)):
        problems.append("object names/order differ from official roster")
    if not trajectory.world_valid.all():
        problems.append("non-finite world flagged invalid")
    if not np.allclose(np.linalg.norm(trajectory.pose_xyzw[:, 3:], axis=-1), 1, atol=1e-3):
        problems.append("non-unit quaternions")
    hashes = provenance(trajectory)
    for field in ("checkpoint_sha256", "eval_script_sha256", "eval_code_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", hashes.get(field, "")):
            problems.append(f"missing/invalid {field}")
    if checkpoint and hashes.get("checkpoint_sha256") != checkpoint["sha256"]:
        problems.append("recorded checkpoint hash disagrees with actual checkpoint")
    if "time_s" not in table.column_names:
        problems.append("missing explicit time_s audit column")
    else:
        times = np.asarray(table["time_s"])
        # These released five source episodes are 20Hz; infer exact value from known source metadata.
        if not np.allclose(times, trajectory.frame_index / expected["fps"], atol=1e-6, rtol=0):
            problems.append("recording timestamps do not preserve source duration")
    return dict(path=str(path), sha256=digest(path), rows=len(actual), schema=trajectory.schema_version,
                provenance=hashes, object_names=expected["object_names"], problems=problems,
                public_schema_valid=not problems,
                official_physics_compatibility="not established by passing a parquet schema check")


def reference_info(directory, expected):
    meta = read(directory / "manifest.json")
    with np.load(directory / "object_poses.npz", allow_pickle=False) as poses:
        names = poses["names"].astype(str).tolist()
        frames = len(poses["world_from_object"])
    return dict(directory=str(directory), frames=frames, fps=float(meta["fps"]), object_names=names,
                matches_roster=names == expected["object_names"] and frames == len(expected["frames"]),
                reference_sha256=digest(directory / "reference.parquet"),
                object_poses_sha256=digest(directory / "object_poses.npz"),
                meshes={name: dict(path=str(directory / "assets" / f"{name}.obj"),
                                   exists=(directory / "assets" / f"{name}.obj").is_file(),
                                   sha256=digest(directory / "assets" / f"{name}.obj") if (directory / "assets" / f"{name}.obj").is_file() else None)
                        for name in expected["object_names"]},
                source_mesh_archive=meta.get("mesh_archive"),
                own_reconstructed_mesh_provenance="not established by archive filename; document reconstruction source")


def audit_selected(entry, expected, kit):
    blockers = []
    required = ("checkpoint", "training_evidence", "evaluation_parquet", "evaluation_report", "evaluator_script", "mesh_provenance", "runtime_reproduction_evidence")
    for key in required:
        if not entry.get(key) or not Path(entry[key]).is_file():
            blockers.append(f"missing real {key} artifact")
    result = dict(blockers=blockers, ready_for_packaging_review=False)
    if blockers:
        return result
    try:
        ck = checkpoint_info(entry["checkpoint"])
        tr = trajectory_info(entry["evaluation_parquet"], expected, kit, ck)
        result.update(checkpoint=ck, trajectory=tr)
        blockers.extend(tr["problems"])
        if not ck["supported_official_algorithm"] or ck["environment_steps"] <= 0:
            blockers.append("checkpoint lacks a supported, trained learner metadata contract")
        if tr["provenance"]["eval_script_sha256"] != digest(entry["evaluator_script"]):
            blockers.append("actual producing evaluator script does not match recorded hash")
        files = entry.get("evaluation_code_files", {})
        if not files:
            blockers.append("actual evaluation module files missing; cannot verify eval_code_sha256")
        else:
            h = hashlib.sha256()
            for name, file in sorted(files.items()):
                h.update(name.encode()); h.update(b"\0"); h.update(Path(file).read_bytes())
            if h.hexdigest() != tr["provenance"]["eval_code_sha256"]:
                blockers.append("actual producing evaluation modules do not match recorded hash")
        evaluation = read(entry["evaluation_report"])
        if evaluation.get("reset_mode") != "explicit" or evaluation.get("start_frame") != 0:
            blockers.append("requires explicit reset at original frame zero")
        if evaluation.get("step_count") != len(expected["frames"]) or evaluation.get("world_count") != 1:
            blockers.append("evaluation report does not cover required frames/one submitted world")
        if evaluation.get("non_finite_worlds"):
            blockers.append("evaluation contains non-finite worlds")
        if evaluation.get("checkpoint_sha256") != ck["sha256"]:
            blockers.append("evaluation report checkpoint hash mismatch")
        # A file's existence does not certify its scientific provenance or policy validity.
        result["manual_review_required"] = ["training evidence describes actual learner updates", "mesh provenance establishes own reconstructed geometry", "runtime reproduction confirms no VOC/object assistance and correct controller/asset semantics"]
        result["structural_checks_pass"] = not blockers
        result["ready_for_packaging_review"] = not blockers
    except (ValueError, KeyError, IndexError, OSError) as error:
        blockers.append(f"{type(error).__name__}: {error}")
    return result


def audit(args):
    local, current = kit_info(args.kit), kit_info(args.current_kit)
    roster = public_roster(args.current_kit)
    policy_manifest = read(args.policy_manifest) if args.policy_manifest else {}
    selected = [int(e) for e in args.episodes]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(roster):
        raise ValueError("Select unique episodes from the official roster")
    entries = []
    historical = REPO / "challenge_submissions/track3/transfer_hawor_chunk1s_20261009"
    for episode in selected:
        name = f"episode_{episode:06d}"
        source = args.references / name
        ref = reference_info(source, roster[episode])
        expected = dict(roster[episode], fps=ref["fps"])
        item = dict(episode=episode, reference=ref,
                    selected_policy=audit_selected(policy_manifest.get("episodes", {}).get(str(episode), {}), expected, args.current_kit))
        old_checkpoint = historical / name / "training/policy_50331648.safetensors"
        old_evaluation = historical / "results" / f"{name}.parquet"
        if old_checkpoint.is_file():
            old = checkpoint_info(old_checkpoint)
            item["historical_candidate"] = dict(checkpoint=old, selected=False,
                current_method=False, reference_matches_current=old["reference_sha256"] == ref["reference_sha256"],
                note="Old learned policy, not a DexAgent refinement result; runtime/physics replay not performed")
            if old_evaluation.is_file():
                item["historical_candidate"]["evaluation"] = trajectory_info(old_evaluation, expected, args.current_kit, old)
        entries.append(item)
    commit = policy_manifest.get("code_commit_url", "")
    commit_valid = bool(re.fullmatch(r"https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}", commit))
    blockers = []
    if local["format"] != current["format"]:
        blockers.append("User-provided kit is older: use reviewed current kit for final combined Track3 artifact")
    if current["manifest_file_mismatches"]:
        blockers.append("Current kit fails its own file-hash manifest")
    if not commit_valid:
        blockers.append("Missing producing GitHub commit URL for the new experiment code")
    if any(not row["selected_policy"]["ready_for_packaging_review"] for row in entries):
        blockers.append("Selected episodes lack audited learned-policy/evaluation/provenance artifacts")
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), offline=True, upload_attempted=False,
                requested_episodes=selected, full_roster=current["required_episodes"],
                outside_current_scope=sorted(set(current["required_episodes"]) - set(selected)),
                subset_only=True, final_upload_ready=False, research_subset_artifacts_ready_for_review=not blockers,
                official_score_available_locally=False, blockers=blockers, local_kit=local, current_kit=current,
                sources={"faq": FAQ_URL, "submission_kit": KIT_URL}, episodes=entries,
                physics_rules=dict(embodiment="FAQ explicitly permits floating hands; Sharpa Wave default",
                    fixed_mass_or_pd_rule_found=False,
                    explanation="No exact fixed mass, stiffness, damping, or collision constants were found in supplied/current kit or FAQ. Checkpoint/runtime configs are defaults, not an organizer approval of every change.",
                    evaluator="Saved embodiment/action/sim/collision config; deterministic full sequence, explicit reset, no auto-reset, VOC scale and force/torque disabled",
                    policy="FAQ requires learned policies; included evaluator supports FlashSAC/PPO typed checkpoints. BC is not explicitly prohibited, but custom BC/runtime compatibility is not certified by this audit."))


def prepare_skeleton(out, episodes):
    for directory in ("prepared/results", "prepared/meshes", "prepared/policy_evidence"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        (out / "prepared/meshes" / f"episode_{episode:06d}").mkdir(exist_ok=True)
    template = out / "policy_manifest.template.json"
    if not template.exists():
        write(template, dict(code_commit_url=None, episodes={str(episode): dict(
            checkpoint=None, training_evidence=None, evaluation_parquet=None, evaluation_report=None,
            evaluator_script=None, evaluation_code_files={}, mesh_provenance=None, runtime_reproduction_evidence=None)
            for episode in episodes}))
    (out / "prepared/NOT_A_SUBMISSION.txt").write_text(
        "EMPTY PREPARATION SKELETON. No synthetic trajectories, placeholder policy weights, or invented hashes.\n"
        "Current scope is episodes 1,3,5,8,10 only. Official full submission requires all 20.\n"
        "Use real learned-policy evaluator artifacts and the current kit packer only after readiness review.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--current-kit", type=Path, default=DEFAULT_CURRENT_KIT)
    parser.add_argument("--references", type=Path, default=REPO / "challenge_reconstruction/final_recon")
    parser.add_argument("--episodes", nargs="+", type=int, default=EPISODES)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = audit(args)
    prepare_skeleton(args.output, args.episodes)
    write(args.output / "readiness.json", report)
    print(json.dumps({key: report[key] for key in ("requested_episodes", "final_upload_ready", "research_subset_artifacts_ready_for_review", "blockers")}, indent=2))
    # Deliberately fail when missing evidence: no packer or uploader is invoked.
    raise SystemExit(0 if report["research_subset_artifacts_ready_for_review"] else 2)


if __name__ == "__main__":
    main()
