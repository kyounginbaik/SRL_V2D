#!/usr/bin/env python3
"""Run the twelve unchanged legacy policies, with bounded CPU concurrency.

No retries or overwrite: an interrupted/failed execution remains available for
inspection. Reruns require a new output root. This runner never uses a GPU.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from common_evaluator import sha, utc, verify_build


EPISODES = [1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 38]


def run(args):
    if not 1 <= args.workers <= 4:
        raise ValueError("Use between one and four concurrent CPU evaluations")
    verify_build(args.build_manifest)
    args.output_root = args.output_root.resolve()
    script = Path(__file__).resolve().with_name("common_evaluator.py")
    status_path = args.output_root / "legacy_batch_status.json"
    logs = args.output_root / "legacy_logs"
    if status_path.exists() or logs.exists():
        raise FileExistsError("Preserve previous batch: " + str(args.output_root))
    for episode in EPISODES:
        name = f"episode_{episode:06d}"
        if (args.output_root / name).exists():
            raise FileExistsError("Refusing to retry/overwrite: " + str(args.output_root / name))
        if not (args.legacy_root / name / "component/portable/bundle_manifest.json").is_file():
            raise FileNotFoundError(args.legacy_root / name)
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir()
    states = {f"episode_{ep:06d}": dict(stage="pending", episode=ep) for ep in EPISODES}
    lock = threading.RLock()
    stopping = threading.Event()
    processes = {}
    started = utc()

    def save():
        with lock:
            snapshot = copy.deepcopy(states)
        completed = sum(state["stage"] == "completed" for state in snapshot.values())
        failed = sum(state["stage"] in ("failed", "cancelled") for state in snapshot.values())
        report = dict(schema="legacy_common_evaluation_batch_v1", started_utc=started,
            updated_utc=utc(), build_manifest_sha256=sha(args.build_manifest),
            workers=args.workers, requested_episodes=EPISODES, completed=completed,
            failed=failed, all_completed=completed == len(EPISODES), episodes=snapshot)
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, status_path)
        print(json.dumps(dict(status=str(status_path), completed=completed, failed=failed)), flush=True)

    def stop(signum, _frame):
        stopping.set()
        with lock:
            children = list(processes.values())
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def one(episode):
        name = f"episode_{episode:06d}"
        if stopping.is_set():
            with lock:
                states[name].update(stage="cancelled", completed_utc=utc())
            return
        output = args.output_root / name
        log = logs / (name + ".log")
        command = [sys.executable, str(script), "evaluate", "--backend", "legacy",
            "--build-manifest", str(args.build_manifest.resolve()), "--kit", str(args.kit.resolve()),
            "--episode", str(episode), "--legacy-component", str((args.legacy_root / name / "component").resolve()),
            "--output", str(output)]
        with lock:
            states[name].update(stage="executing_policy_and_independent_replay", started_utc=utc(),
                                log=str(log), command=command)
        try:
            with log.open("w") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1",
                    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                with lock:
                    processes[name] = child
                    states[name]["pid"] = child.pid
                if stopping.is_set():
                    os.killpg(child.pid, signal.SIGTERM)
                returncode = child.wait()
            done = output / "completed.json"
            if returncode == 0 and not done.is_file():
                raise RuntimeError("Evaluation exited successfully without completed.json")
            with lock:
                states[name].update(stage="completed" if returncode == 0 else "failed",
                    completed_utc=utc(), returncode=returncode,
                    completed_manifest=str(done) if done.is_file() else None)
        except Exception as error:
            with lock:
                states[name].update(stage="failed", completed_utc=utc(), error=str(error))
        finally:
            with lock:
                processes.pop(name, None)

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(one, episode) for episode in EPISODES}
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            save()
    save()
    return 0 if all(value["stage"] == "completed" for value in states.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("build-manifest", "kit", "output-root", "legacy-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    raise SystemExit(run(p.parse_args()))
