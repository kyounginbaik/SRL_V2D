#!/usr/bin/env python3
"""Fresh CPU baselines and released DexAgent MPC for the seven HaWoR episodes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from prepare_seven import BASE, ROOT, LEGACY, EPISODES, sha, write
sys.path.insert(0, str(LEGACY / "code"))
sys.path.insert(0, str(ROOT / "third_party/DexAgent"))
from floating_runtime import _freeze, execute_reference
from calibrate_controller import controller_scene
from floating_tracking import refine
from evaluate_mp_sr import evaluate_files


def evaluate(input_dir, scene, reference, output):
    manifest = json.loads((input_dir / "input_manifest.json").read_text())
    execution = execute_reference(scene, reference, output / "execution",
                                  expected_duration_s=(manifest["frames"] - 1) / manifest["fps"])
    metrics = evaluate_files(manifest["source"], execution["trajectory"], input_dir / "input_manifest.json",
                             execution["scene"], input_dir / "metric_geometry.json")
    return execution, metrics


def run(episode, tag, samples, workers, max_minutes, seed, baseline_only=False):
    if episode not in EPISODES:
        raise ValueError("Only the seven new episodes are permitted")
    name = f"episode_{episode:06d}"
    input_dir = BASE / "episodes" / name
    if not (input_dir / "preparation_complete.json").exists():
        raise FileNotFoundError("Input preparation has not completed: " + str(input_dir))
    root = BASE / "runs" / tag / name
    root.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    manifest = json.loads((input_dir / "input_manifest.json").read_text())
    reference = input_dir / "tracking_reference.npz"
    begin = time.monotonic()
    baseline = root / "baseline"
    if (baseline / "result.json").exists():
        base_record = json.loads((baseline / "result.json").read_text())
        if base_record["reference_sha256"] != sha(reference):
            raise ValueError("Baseline belongs to different reference")
    else:
        write(status, dict(episode=name, stage="full_horizon_baseline"))
        baseline.mkdir(exist_ok=True)
        frozen_dir = baseline / "frozen_initial"
        frozen_dir.mkdir(exist_ok=True)
        _, frozen = _freeze(input_dir / "stage2/scene.xml", frozen_dir)
        scene = baseline / "controller_scene.xml"
        controller = controller_scene(frozen, scene, "kp5000_r200_d5")
        execution, metrics = evaluate(input_dir, scene, reference, baseline)
        base_record = dict(episode=name, input_dir=str(input_dir), learned_policy=False,
                           method="HaWoR/Sharpa source reference in bounded unassisted MuJoCo physics",
                           controller=controller, reference=str(reference), reference_sha256=sha(reference),
                           execution=execution, metrics=metrics, wall_seconds=time.monotonic() - begin,
                           original_object_targets_unchanged=True, submission_ready=False)
        write(baseline / "result.json", base_record)
        print(json.dumps(dict(episode=name, stage="baseline_complete",
                              mp_sr=metrics["reference_proxy_mp_sr"], wall_seconds=base_record["wall_seconds"])), flush=True)
    if baseline_only:
        write(status, dict(episode=name, stage="baseline_complete", mp_sr=base_record["metrics"]["reference_proxy_mp_sr"]))
        return base_record
    output = root / "native_mpc"
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    output.mkdir(exist_ok=True)
    if not base_record["execution"]["physics_verified"]:
        raise ValueError("Baseline independent physics replay failed")
    task = output / "tracking_task.json"
    write(task, dict(scope="Track every source object over unchanged original timeline",
                     subgoals=[dict(index=0, motion="track_reference", acting_hand="both",
                                    object_refs=[item["id"] for item in manifest["objects"]])]))
    write(status, dict(episode=name, stage="released_DexAgent_MPC", samples=samples, workers=workers,
                      max_minutes=max_minutes, baseline_mp_sr=base_record["metrics"]["reference_proxy_mp_sr"]))
    search_begin = time.monotonic()
    search = refine(workspace=str(output), scene_semantic_json=str(task),
                    scene_recon_json=str(input_dir / "stage2/scene_recon.json"),
                    reference=str(reference), scene=base_record["execution"]["scene"],
                    out_dir=str(output / "search"), samples=samples, workers=workers,
                    max_minutes=max_minutes, seed=seed)
    write(status, dict(episode=name, stage="MPC_full_horizon_replay"))
    execution, metrics = evaluate(input_dir, base_record["execution"]["scene"], search["reference"], output)
    result = dict(episode=name, input_dir=str(input_dir),
                  method="Released DexAgent predictive sampling MPC, floating-Sharpa runtime port",
                  upstream_commit="e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633",
                  native_full_LLM_pipeline=False, learned_policy=False, search=search,
                  execution=execution, metrics=metrics, baseline_metrics=base_record["metrics"],
                  scene_sha256=sha(execution["scene"]), reference_sha256=sha(reference),
                  original_object_targets_unchanged=True, original_timing_unchanged=True,
                  search_and_replay_seconds=time.monotonic() - search_begin,
                  wall_seconds=time.monotonic() - begin, submission_ready=False)
    write(output / "result.json", result)
    write(status, dict(episode=name, stage="complete", reference_proxy_mp_sr=metrics["reference_proxy_mp_sr"],
                       baseline_proxy_mp_sr=base_record["metrics"]["reference_proxy_mp_sr"],
                       elapsed_seconds=result["wall_seconds"]))
    print(json.dumps(dict(episode=name, mp_sr=metrics["reference_proxy_mp_sr"], wall_seconds=result["wall_seconds"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--tag", default="hawor_mpc_v1")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--baseline-only", action="store_true")
    run(**vars(parser.parse_args()))
