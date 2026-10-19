#!/usr/bin/env python3
"""Actually execute either preserved controller backend under one audited build.

This wrapper never takes an existing rollout as the result of a new evaluation.
It launches the pinned backend into a new directory, validates its fresh physical
recording, and exports that result. Original backend exports/provenance survive.
The common digest includes both implementations, not only this wrapper.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".sh"}


def source_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and
                  p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts and ".git" not in p.parts)


def source_pair(value):
    label, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("Expected a simple label=/absolute/path source argument")
    return label, Path(path).resolve()


def make_manifest(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not re.fullmatch(r"https://[\w.-]*github[\w.-]*/[\w.-]+/[\w.-]+/commit/[0-9a-fA-F]{7,40}/?", args.code_commit_url):
        raise ValueError("Provide the actual producing GitHub commit URL")
    roots = [source_pair(v) for v in args.source]
    roots.append(("common_package", Path(__file__).resolve().parent))
    if len({name for name, _ in roots}) != len(roots):
        raise ValueError("Duplicate source labels")
    records = []
    for label, root in sorted(roots):
        files = source_files(root) if root.is_dir() else [root]
        if not files:
            raise ValueError("Empty source root: " + str(root))
        for path in files:
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root) if root.is_dir() else Path(root.name)
            records.append(dict(key=str(Path(label) / relative), path=str(path), sha256=sha(path)))
    digest = hashlib.sha256()
    for entry in records:
        digest.update(Path(entry["path"]).read_bytes())
    manifest = dict(schema="uniretarget_mixed_evaluator_build_v1", created_utc=utc(),
        code_commit_url=args.code_commit_url, evaluator_script=str(Path(__file__).resolve()),
        eval_script_sha256=sha(__file__), eval_code_sha256=digest.hexdigest(),
        digest_convention="SHA256 of concatenated source bytes in the recorded ordered list",
        sources=records, source_roots={label: str(path) for label, path in roots},
        backend_kinds=["frozen_mujoco_rbf", "frozen_uniretarget_ilchold"],
        official_runtime_certification=False)
    write(args.output, manifest)
    print(json.dumps(dict(manifest=str(args.output), files=len(records), eval_code_sha256=digest.hexdigest())))


def verify_build(path):
    manifest = read(path)
    if manifest["schema"] != "uniretarget_mixed_evaluator_build_v1":
        raise ValueError("Unknown common evaluator manifest")
    digest = hashlib.sha256()
    found = set()
    for entry in manifest["sources"]:
        if entry["key"] in found or sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Changed/duplicate actual source: " + entry["key"])
        found.add(entry["key"])
        digest.update(Path(entry["path"]).read_bytes())
    if digest.hexdigest() != manifest["eval_code_sha256"] or sha(__file__) != manifest["eval_script_sha256"]:
        raise ValueError("Common evaluator source differs from the frozen build")
    return manifest


def require_recorded_source(manifest, path):
    path = Path(path).resolve()
    entries = [entry for entry in manifest["sources"] if Path(entry["path"]).resolve() == path]
    if not entries or any(entry["sha256"] != sha(path) for entry in entries):
        raise ValueError("Executed source is not included in the common build: " + str(path))


def verify_equivalent_tree(manifest, label, directory):
    """Legacy bundles repeat identical frozen code; check each executed copy."""
    directory = Path(directory)
    prefix = label + "/"
    expected = {e["key"][len(prefix):]: e["sha256"] for e in manifest["sources"] if e["key"].startswith(prefix)}
    actual = {str(p.relative_to(directory)): sha(p) for p in source_files(directory)}
    if not expected or expected != actual:
        raise ValueError("Executed backend tree differs from frozen " + label)


def official_validate(parquet, kit, episode, checkpoint_sha256):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    sys.path.insert(0, str(kit))
    from v2dlb.submission_parquet import read_trajectory, provenance
    result = read_trajectory(parquet)
    sample = pd.read_parquet(kit / "data/track_3_sample_submission.parquet", columns=["row_id"])
    wanted = sample[sample.row_id.str.startswith(f"track_3/e{episode:06d}/")].row_id.tolist()
    got = [f"track_3/e{episode:06d}/w{int(w):03d}/f{int(f):06d}/o{int(o)}"
           for w, f, o in zip(result.world_index, result.frame_index, result.object_slot)]
    if result.episode_index != episode or sorted(got) != sorted(wanted) or len(set(got)) != len(got):
        raise ValueError("Export does not exactly match the unchanged public episode grid")
    names = read(kit / "data/track_3_evaluation_objects.json")["episodes"][str(episode)]
    if result.object_name is None or any(str(n) != names[int(s)] for n, s in zip(result.object_name, result.object_slot)):
        raise ValueError("Incorrect public object names/order")
    if not result.world_valid.all() or not np.isfinite(result.pose_xyzw).all():
        raise ValueError("Invalid physical world")
    if not np.allclose(np.linalg.norm(result.pose_xyzw[:, 3:], axis=-1), 1, atol=1e-3):
        raise ValueError("Non-unit quaternion")
    table = pq.read_table(parquet)
    if "time_s" not in table.column_names or not np.allclose(np.asarray(table["time_s"]), result.frame_index / 20., rtol=0, atol=1e-6):
        raise ValueError("Source 20 Hz timestamp grid is not preserved")
    prov = provenance(result)
    if prov.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Exported checkpoint lineage is inconsistent")
    for name in ("checkpoint_sha256", "eval_code_sha256", "eval_script_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", prov.get(name, "")):
            raise ValueError("Missing actual producing provenance: " + name)
    return dict(rows=len(got), object_names=names, full_public_grid=True,
        source_timestamp_grid=True, finite=True, provenance=prov,
        sha256=sha(parquet), official_runtime_certification=False)


def validate_meshes(meshes, kit, episode, names):
    import numpy as np
    import pandas as pd
    from types import ModuleType
    sys.path.insert(0, str(kit))
    namespace = ModuleType("tools")
    namespace.__path__ = [str(kit / "tools")]
    sys.modules["tools"] = namespace
    from v2dlb.track3_pose import pack_object_meshes, validate_packed_meshes
    frame = pd.DataFrame({"row_id": [f"track_3/e{episode:06d}/w000/f000000/o{i}" for i in range(len(names))]})
    decoded = validate_packed_meshes(pack_object_meshes(frame, meshes, {str(episode): names}))
    return {names[slot]: dict(vertices=len(v), faces=len(f), finite=bool(np.isfinite(v).all()))
            for (_, slot), (v, f) in decoded.items()}


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fresh_legacy(args, build):
    import numpy as np
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    component = args.legacy_component.resolve()
    bundle = component / "portable"
    verify_equivalent_tree(build, "legacy_code", bundle / "code")
    verify_equivalent_tree(build, "legacy_dexagent", bundle / "third_party/DexAgent")
    meta = read(bundle / "bundle_manifest.json")
    if meta["episode_id"] != args.episode or args.episode == 37:
        raise ValueError("Legacy backend episode mismatch")
    command = [sys.executable, str(bundle / "code/reproduce_policy_component.py"), "replay",
        "--bundle", str(bundle), "--output", str(args.output / "backend")]
    subprocess.run(command, check=True, cwd="/tmp")
    validation = read(args.output / "backend/portable_validation.json")
    for key in ("all_physical_states_actions_exact", "compiled_model_arrays_options_exact",
                "original_weights_unchanged", "source_object_targets_unchanged"):
        if not validation[key]:
            raise ValueError("Fresh legacy replay failed: " + key)
    if not validation["independent_control_replay"]["ok"] or validation["reference_proxy_mp_sr"] != 1:
        raise ValueError("Fresh legacy physical execution/proxy score failed")
    result = read(args.output / "backend/evaluation/result.json")
    sys.path.insert(0, str(bundle / "code"))
    metric = module_at("common_legacy_metrics", bundle / "code/evaluate_mp_sr.py")
    source = read(bundle / "original/input_manifest.json")
    achieved = metric.apply_world_transform(metric.load_trajectory(result["execution"]["trajectory"]),
                                            np.asarray(source["source_from_setup"]))
    reference = metric.load_reference(bundle / "source" / meta["episode"])
    if not np.allclose(achieved["timestamps_seconds"], reference["timestamps_seconds"], rtol=0, atol=1e-6):
        raise ValueError("Fresh evaluation timestamps changed")
    writer_path = Path(build["source_roots"]["schema_writer"])
    require_recorded_source(build, writer_path)
    writer = module_at("common_schema_writer", writer_path)
    poses = lambda a: np.concatenate([a["object_positions"], a["object_quaternions_wxyz"][..., [1, 2, 3, 0]]], axis=-1)
    raw = args.output / "backend/fresh_trajectory.parquet"
    writer.write_object_trajectories_parquet(raw, SimpleNamespace(achieved_pose_w=poses(achieved),
        reference_pose_w=poses(reference)[:, 0], object_body_names=achieved["object_names"],
        body_object_ids=np.arange(len(achieved["object_names"])), non_finite_worlds=(), start_frame=0, control_fps=20.),
        checkpoint_sha256=meta["original_checkpoint_sha256"],
        eval_script_sha256=sha(bundle / "code/evaluate_policy_component.py"),
        eval_code_sha256=read(component / "readiness.json")["evaluation_code_sha256"],
        reference_parquet=bundle / "source" / meta["episode"] / "reference.parquet")
    # The old export is only an independent comparison, never the new output.
    old = pq.read_table(component / "results" / (meta["episode"] + ".parquet"))
    new = pq.read_table(raw)
    for name in old.column_names:
        if not np.array_equal(np.asarray(old[name]), np.asarray(new[name])):
            raise ValueError("Fresh exported pose/row data differs from reviewed recording: " + name)
    return dict(raw_parquet=raw, checkpoint=bundle / "original/policy.npz",
        actions=Path(result["execution"]["trajectory"]), metrics=result["metrics"],
        backend_command=command, backend_kind="frozen_mujoco_rbf", legacy_component=str(component),
        source_reference_sha256=meta["source_reference_sha256"],
        exact_previous_physical_states_actions=True,
        mesh_source=component / "meshes" / meta["episode"])


def fresh_ilchold(args, build):
    import numpy as np
    require_recorded_source(build, args.exporter)
    command = [sys.executable, str(args.exporter.resolve()), "--run-root", str(args.run_root.resolve()),
        "--output", str(args.output / "backend"), "--world-count", str(args.world_count),
        "--world-index", str(args.world_index)]
    subprocess.run(command, check=True)
    report = read(args.output / "backend/report.json")
    raw = args.output / "backend/episode_000037.parquet"
    checkpoint = args.run_root / "runs/pilot/episode_000037/chunk/seed_44/training/policy_2508800.safetensors"
    if report.get("checkpoint_sha256") != sha(checkpoint):
        raise ValueError("Fresh ilchold backend checkpoint differs")
    actions = args.output / "backend/actions.npz"
    if not actions.is_file() or not raw.is_file():
        raise ValueError("Fresh ilchold backend did not write actions and actual poses")
    expected = dict(frame_count=151, fps=20., physical_transitions=150,
                    selected_world_index=args.world_index, world_count=args.world_count,
                    no_object_assistance=True, voc_scale=0., voc_max_force=0., voc_max_torque=0.,
                    automatic_reset=False, selected_world_strict_proxy_pass=True)
    for key, value in expected.items():
        if key not in report or report[key] != value:
            raise ValueError("Ilchold physical/proxy contract failed: " + key)
    if not report.get("actions_semantics"):
        raise ValueError("Ilchold action/state convention is not documented")
    reference = args.run_root / "frozen/episode_000037/reference.parquet"
    refined = args.run_root / "runs/pilot/episode_000037/chunk/seed_44/training/references/reference_0005.npz"
    if (report.get("source_reference_sha256") != sha(reference) or
            report.get("reference_refinement_sha256") != sha(refined)):
        raise ValueError("Ilchold source/refined-reference lineage mismatch")
    with np.load(actions, allow_pickle=False) as recording:
        for key in recording.files:
            array = recording[key]
            if array.dtype.kind in "biufc" and not np.isfinite(array).all():
                raise ValueError("Non-finite actual ilchold recording: " + key)
        for key, rows in [("body_q", 151), ("joint_q", 151), ("policy_action", 150),
                          ("joint_target", 150), ("time_s", 151), ("action_time_s", 150)]:
            if key not in recording or recording[key].shape[0] != rows:
                raise ValueError("Ilchold causal action/state recording shape failed: " + key)
        if recording["policy_action"].shape != (150, 56):
            raise ValueError("Expected 150 actual 56-dimensional policy actions")
        if np.any(recording["object_assistance_scale"] != 0):
            raise ValueError("Recorded object assistance is nonzero")
        if not np.allclose(recording["time_s"], np.arange(151) / 20., rtol=0, atol=1e-9):
            raise ValueError("Ilchold full original time grid changed")
    strict = report["strict_source_reference_metrics_all_151_frames"]
    if strict["strict_mp_sr_per_world"][args.world_index] is not True:
        raise ValueError("Selected video world failed the strict full-frame source-reference proxy")
    return dict(raw_parquet=raw, checkpoint=checkpoint, actions=actions, metrics=report,
        backend_command=command, backend_kind="frozen_uniretarget_ilchold",
        source_reference_sha256=sha(args.run_root / "frozen/episode_000037/reference.parquet"),
        refined_reference_sha256=sha(args.run_root / "runs/pilot/episode_000037/chunk/seed_44/training/references/reference_0005.npz"),
        selected_world_index=args.world_index, world_count=args.world_count,
        mesh_source=None)


def evaluate(args):
    import pyarrow.parquet as pq
    build = verify_build(args.build_manifest)
    if args.output.exists():
        raise FileExistsError("Preserve existing evaluation: " + str(args.output))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    started = utc()
    if args.backend == "legacy":
        result = fresh_legacy(args, build)
    else:
        args.episode = 37
        result = fresh_ilchold(args, build)
    verify_build(args.build_manifest)
    checkpoint_hash = sha(result["checkpoint"])
    raw_validation = official_validate(result["raw_parquet"], args.kit, args.episode, checkpoint_hash)
    name = f"episode_{args.episode:06d}"
    parquet = args.output / "results" / (name + ".parquet")
    parquet.parent.mkdir()
    table = pq.read_table(result["raw_parquet"])
    metadata = dict(table.schema.metadata or {})
    # These identify the wrapper that just executed both the runtime and export.
    # The untouched backend export remains evidence of its original attribution.
    metadata.update({b"flash_chord.eval_script_sha256": build["eval_script_sha256"].encode(),
        b"flash_chord.eval_code_sha256": build["eval_code_sha256"].encode(),
        b"participant.common_build_manifest_sha256": sha(args.build_manifest).encode(),
        b"participant.original_backend_export_sha256": sha(result["raw_parquet"]).encode(),
        b"participant.evaluator": b"common dispatcher with freshly executed, pinned physical backend",
        b"participant.backend": result["backend_kind"].encode(),
        b"participant.code_commit_url": build["code_commit_url"].encode()})
    pq.write_table(table.replace_schema_metadata(metadata), parquet)
    validation = official_validate(parquet, args.kit, args.episode, checkpoint_hash)
    meshes = args.output / "meshes" / name
    if result["mesh_source"] is not None:
        shutil.copytree(result["mesh_source"], meshes)
    else:
        meshes.mkdir(parents=True)
        supplied = dict(source_pair(v) for v in args.mesh)
        if set(supplied) != set(validation["object_names"]):
            raise ValueError("Supply the actual rollout body-frame mesh for every object")
        for object_name, path in supplied.items():
            shutil.copy2(path, meshes / (object_name + path.suffix))
    mesh_validation = validate_meshes(args.output / "meshes", args.kit, args.episode, validation["object_names"])
    record = {key: value for key, value in result.items() if key not in ("mesh_source", "raw_parquet", "checkpoint", "actions")}
    record.update(schema="uniretarget_common_episode_execution_v1", episode=name,
        started_utc=started, completed_utc=utc(), build_manifest_sha256=sha(args.build_manifest),
        eval_code_sha256=build["eval_code_sha256"], eval_script_sha256=build["eval_script_sha256"],
        code_commit_url=build["code_commit_url"],
        checkpoint=dict(path=str(result["checkpoint"]), sha256=checkpoint_hash),
        actions=dict(path=str(result["actions"]), sha256=sha(result["actions"])),
        backend_export=dict(path=str(result["raw_parquet"]), sha256=sha(result["raw_parquet"]), validation=raw_validation),
        trajectory=dict(path=str(parquet), sha256=sha(parquet), validation=validation),
        meshes={p.stem: dict(path=str(p), sha256=sha(p)) for p in sorted(meshes.iterdir()) if p.is_file()},
        mesh_validation=mesh_validation, fresh_backend_executed=True,
        original_backend_evidence_preserved=True, source_reference_proxy_not_official_gt=True,
        official_runtime_certification=False, complete_official_submission_ready=False, uploaded=False)
    shutil.copy2(args.build_manifest, args.output / "build_manifest.json")
    write(args.output / "completed.json", record)
    print(json.dumps(dict(completed=str(args.output / "completed.json"), rows=validation["rows"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze = modes.add_parser("build-manifest")
    freeze.add_argument("--source", action="append", default=[], help="label=directory (Python/config/shell sources) or label=file")
    freeze.add_argument("--code-commit-url", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("evaluate")
    for key in ("build-manifest", "output", "kit"):
        run.add_argument("--" + key, type=Path, required=True)
    run.add_argument("--backend", choices=["legacy", "ilchold"], required=True)
    run.add_argument("--episode", type=int)
    run.add_argument("--legacy-component", type=Path)
    run.add_argument("--exporter", type=Path)
    run.add_argument("--run-root", type=Path)
    run.add_argument("--world-count", type=int, default=32)
    run.add_argument("--world-index", type=int, default=28)
    run.add_argument("--mesh", action="append", default=[])
    args = parser.parse_args()
    if args.mode == "build-manifest":
        make_manifest(args)
    else:
        if args.backend == "legacy" and (args.episode is None or args.legacy_component is None):
            parser.error("Legacy execution requires --episode and --legacy-component")
        if args.backend == "ilchold" and (args.exporter is None or args.run_root is None):
            parser.error("Ilchold execution requires --exporter and --run-root")
        evaluate(args)


if __name__ == "__main__":
    main()
