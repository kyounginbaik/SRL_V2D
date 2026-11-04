#!/usr/bin/env python3
"""Summarize real baseline/refined DexAgent runs; never ingest reference self-checks.

Only runs/TAG/episode_NNNNNN/seed_SEED/result.json supplies completed results.
Missing/error statuses remain visible and never silently disappear from a denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = (1, 3, 5, 8, 10)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean_or_none(values):
    values = list(values)
    return float(mean(values)) if values and all(finite(x) for x in values) else None


def read_json(path):
    return json.loads(Path(path).read_text())


def reference_changes(search):
    """Measure the selected artifact, not the rejected candidate or random samples."""
    import numpy as np
    before = search.get("reference_in", {}).get("path")
    after = search.get("reference")
    if not before or not after or not Path(before).is_file() or not Path(after).is_file():
        return {"available": False, "reason": "selected/input reference artifact not available"}
    try:
        with np.load(before, allow_pickle=False) as src, np.load(after, allow_pickle=False) as dst:
            names = src["joint_names"].astype(str).tolist()
            dst_names = dst["joint_names"].astype(str).tolist()
            if len(set(names)) != len(names) or len(set(dst_names)) != len(dst_names) or set(names) != set(dst_names):
                raise ValueError("reference coordinate identities differ")
            a, b = np.asarray(src["qpos"], float), np.asarray(dst["qpos"], float)[:, [dst_names.index(n) for n in names]]
            if a.shape != b.shape or float(src["fps"]) != float(dst["fps"]):
                raise ValueError("selected reference duration or dimensions changed")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite command artifact")
        delta = b - a
        robot = [i for i, name in enumerate(names) if name.startswith("robot_")]
        if not robot:
            raise ValueError("no identified robot coordinates")
        changed = np.any(abs(delta[:, robot]) > 1e-7, axis=1)
        hands = {}
        for side in ("left", "right"):
            translation = [next(i for i, name in enumerate(names) if name.endswith(f"{side}_pos_{axis}")) for axis in "xyz"]
            rotation = [next(i for i, name in enumerate(names) if name.endswith(f"{side}_rot_{axis}")) for axis in "xyz"]
            fingers = [i for i in robot if names[i].startswith(f"robot_{side}_") and i not in translation + rotation]
            dp = np.linalg.norm(delta[:, translation], axis=1)
            hands[side] = dict(wrist_translation_mean_cm=float(dp.mean() * 100),
                               wrist_translation_max_cm=float(dp.max() * 100),
                               wrist_rotation_joint_abs_mean_deg=float(np.degrees(abs(delta[:, rotation])).mean()),
                               wrist_rotation_joint_abs_max_deg=float(np.degrees(abs(delta[:, rotation])).max()),
                               finger_joint_abs_mean_deg=float(np.degrees(abs(delta[:, fingers])).mean()) if fingers else None,
                               finger_joint_abs_max_deg=float(np.degrees(abs(delta[:, fingers])).max()) if fingers else None)
        return dict(available=True, input_reference=str(before), selected_reference=str(after),
                    frames=len(a), changed_frames=int(changed.sum()), changed_frame_fraction=float(changed.mean()),
                    change_threshold=1e-7, hands=hands,
                    rotation_note="Differences of commanded wrist rotation joint coordinates, not an SO(3) geodesic",
                    scope="Selected robot reference commands; object target columns excluded")
    except (ValueError, KeyError, IndexError, StopIteration) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def summarize_variant(metrics, execution):
    official = metrics.get("official_style_reference_proxy", {})
    declared = metrics.get("declared_frame_reference_proxy", {})
    raw, fixed = metrics.get("reference_proxy_mp_sr"), metrics.get("declared_frame_mp_sr")
    full = metrics.get("full_horizon") is True
    metric_valid = (metrics.get("estimated") is True and metrics.get("leaderboard_score") is False
                    and full and finite(raw) and 0 <= raw <= 1)
    physics = execution.get("physics_verified") is True
    completed = execution.get("execution_completed") is True
    replay = execution.get("independent_control_replay", {})
    contract = execution.get("runtime_contract", {})
    objects = {}
    for name, item in official.get("objects", {}).items():
        objects[name] = dict(mean_position_error_cm=item.get("mean_position_error_cm"),
                             mean_orientation_error_degrees=item.get("mean_orientation_error_degrees"),
                             success_per_world=item.get("success_per_world"),
                             declared_frame_mean_position_error_cm=declared.get("objects", {}).get(name, {}).get("mean_position_error_cm"),
                             declared_frame_mean_orientation_error_degrees=declared.get("objects", {}).get(name, {}).get("mean_orientation_error_degrees"))
    return dict(reference_proxy_mp_sr=raw, declared_frame_mp_sr=fixed,
                metric_valid=metric_valid, full_horizon=full,
                physics_verified=physics, execution_completed=completed,
                physically_recorded_mp_sr=(float(raw) if physics and completed else 0.) if metric_valid else None,
                independent_replay_ok=replay.get("ok"), runtime_contract_ok=contract.get("ok"),
                physics_issues={"replay": replay.get("issues", []), "contract": contract.get("issues", []), "unstable": execution.get("unstable")},
                duration_s=execution.get("duration_s"), frames=execution.get("frames"),
                max_hand_pen_m=execution.get("max_hand_pen_m"), max_object_pen_m=execution.get("max_object_pen_m"),
                peak_wrist_force_n=execution.get("peak_wrist_force_n"), peak_wrist_torque_nm=execution.get("peak_wrist_torque_nm"),
                objects=objects, trajectory=execution.get("trajectory"), scene=execution.get("scene"))


def collect_episode(workspace, tag, episode, seed):
    name = f"episode_{int(episode):06d}"
    directory = workspace / "runs" / tag / name / f"seed_{seed}"
    result_path, status_path = directory / "result.json", directory / "status.json"
    row = dict(episode=name, seed=seed, directory=str(directory), status="not_started")
    if not result_path.is_file():
        if status_path.is_file():
            status = read_json(status_path)
            row.update(status=status.get("stage", "unknown"), error=status.get("error"),
                       elapsed_seconds=status.get("elapsed_seconds"), started_utc=status.get("started_utc"))
        return row
    data = read_json(result_path)
    if data.get("episode") != name or data.get("seed") != seed:
        raise ValueError(f"Result identity disagrees with path: {result_path}")
    row.update(status=data.get("stage", "unknown"), result=str(result_path))
    if row["status"] != "complete":
        row["error"] = "result.json exists but is not a complete run"
        return row
    search = data.get("search", {})
    before, proposed = search.get("cost_before"), search.get("cost_after")
    retained = search.get("improved") is True
    selected = proposed if retained else before
    drop = before - selected if finite(before) and finite(selected) else None
    row.update(baseline=summarize_variant(data.get("baseline_metrics", {}), data.get("baseline_execution", {})),
               refined=summarize_variant(data.get("refined_metrics", {}), data.get("refined_execution", {})),
               elapsed_seconds=data.get("elapsed_seconds"), started_utc=data.get("started_utc"), completed_utc=data.get("completed_utc"),
               settings=data.get("settings"), native_full_pipeline=data.get("native_full_pipeline"),
               search=dict(ok=search.get("ok"), retained_refinement=retained,
                           baseline_cost=before, proposed_cost=proposed, selected_cost=selected,
                           selected_cost_reduction=drop,
                           selected_cost_reduction_percent=100 * drop / abs(before) if drop is not None and before else None,
                           replans=search.get("replans"), rollouts=search.get("rollouts"),
                           seconds=search.get("seconds"), stopped=search.get("stopped"),
                           coverage=search.get("coverage"), settings=search.get("settings"),
                           cost_note="Upstream physical tracking objective; cost_after is proposed, selected reverts to baseline when rejected"),
               selected_reference_change=reference_changes(search))
    for variant in ("baseline", "refined"):
        if not row[variant]["metric_valid"]:
            row.setdefault("issues", []).append(f"{variant} MP-SR report is missing or invalid")
    return row


def summarize(workspace=DEFAULT_WORKSPACE, tag="pilot", seed=44, episodes=DEFAULT_EPISODES):
    workspace = Path(workspace).resolve()
    episodes = [int(episode) for episode in episodes]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Episode list must be nonempty and unique")
    rows = [collect_episode(workspace, tag, episode, seed) for episode in episodes]
    completed = [row for row in rows if row["status"] == "complete"]
    paired = [row for row in completed if row["baseline"]["metric_valid"] and row["refined"]["metric_valid"]]
    aggregate = dict(requested_episodes=len(rows), completed_episodes=len(completed), valid_paired_episodes=len(paired),
                     all_requested_complete=len(paired) == len(rows),
                     incomplete_episodes=[row["episode"] for row in rows if row not in paired],
                     completed_elapsed_seconds_sum=mean_or_none([row.get("elapsed_seconds") for row in completed]))
    elapsed = [row.get("elapsed_seconds") for row in completed]
    aggregate["completed_elapsed_seconds_sum"] = sum(elapsed) if elapsed and all(finite(x) for x in elapsed) else None
    aggregate["elapsed_note"] = "Sum of per-run durations; parallel jobs overlap, so this is not makespan"
    for variant in ("baseline", "refined"):
        raw = mean_or_none(row[variant]["reference_proxy_mp_sr"] for row in paired)
        physical = mean_or_none(row[variant]["physically_recorded_mp_sr"] for row in paired)
        declared = mean_or_none(row[variant]["declared_frame_mp_sr"] for row in paired)
        aggregate[variant] = dict(completed_paired_mp_sr=raw, completed_paired_physical_mp_sr=physical,
                                 completed_paired_declared_frame_mp_sr=declared,
                                 full_requested_mp_sr=raw if len(paired) == len(rows) else None,
                                 full_requested_physical_mp_sr=physical if len(paired) == len(rows) else None,
                                 physics_verified_episodes=sum(row[variant]["physics_verified"] for row in paired),
                                 completed_execution_episodes=sum(row[variant]["execution_completed"] for row in paired))
    if paired:
        aggregate["paired_mp_sr_change"] = aggregate["refined"]["completed_paired_mp_sr"] - aggregate["baseline"]["completed_paired_mp_sr"]
        aggregate["retained_refinement_episodes"] = sum(row["search"]["retained_refinement"] for row in paired)
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), workspace=str(workspace), tag=tag, seed=seed,
                metric="Reference-proxy MP-SR; reconstructed poses substitute for confidential ground truth",
                leaderboard_score=False, scope="DexAgent predictive-sampling MPC port to floating Sharpa; not native complete DexAgent pipeline",
                aggregation="Equal weight per requested episode; incomplete runs never count as successes or disappear from requested denominator",
                aggregate=aggregate, episodes=rows)


def number(value, digits=2):
    return f"{value:.{digits}f}" if finite(value) else "—"


def per_world(values, digits=2):
    return "/".join(number(value, digits) for value in values) if isinstance(values, list) and values else "—"


def render(report):
    total = report["aggregate"]
    lines = [f"DexAgent floating Sharpa: {report['tag']}, seed {report['seed']}",
             "Reference-proxy MP-SR, not a leaderboard score. Complete execution and independent control replay are checked separately; passing these checks does not certify penetration-free motion.",
             f"Completed {total['completed_episodes']}/{total['requested_episodes']}; valid paired metrics {total['valid_paired_episodes']}.", "",
             "| Episode | Status | MP-SR baseline → refined | Execution/replay baseline/refined | Selected cost change | Search / total min |",
             "|---|---|---|---|---|---|"]
    for row in report["episodes"]:
        if row["status"] != "complete":
            lines.append(f"| {row['episode'][-6:]} | {row['status']} | — | — | — | — |")
            continue
        baseline, refined, search = row["baseline"], row["refined"], row["search"]
        physical = "/".join("pass" if item["physics_verified"] and item["execution_completed"] else "FAIL" for item in (baseline, refined))
        before, selected = search["baseline_cost"], search["selected_cost"]
        search_min = search["seconds"] / 60 if finite(search["seconds"]) else None
        total_min = row["elapsed_seconds"] / 60 if finite(row["elapsed_seconds"]) else None
        lines.append(f"| {row['episode'][-6:]} | complete | {number(baseline['reference_proxy_mp_sr'])} → {number(refined['reference_proxy_mp_sr'])} | {physical} | {number(before)} → {number(selected)} | {number(search_min)} / {number(total_min)} |")
    lines += ["", "Per-object errors below use official-style shared initial-scene alignment; multiple worlds are separated by `/`.", "",
              "| Episode / object | Mean position cm: baseline → refined | Mean orientation deg: baseline → refined |",
              "|---|---|---|"]
    for row in report["episodes"]:
        if row["status"] != "complete":
            continue
        b, r = row["baseline"]["objects"], row["refined"]["objects"]
        for name in sorted(set(b) | set(r)):
            before, after = b.get(name, {}), r.get(name, {})
            lines.append(f"| {row['episode'][-6:]} / {name} | {per_world(before.get('mean_position_error_cm'))} → {per_world(after.get('mean_position_error_cm'))} | {per_world(before.get('mean_orientation_error_degrees'))} → {per_world(after.get('mean_orientation_error_degrees'))} |")
    lines.append("")
    for row in report["episodes"]:
        if row["status"] != "complete":
            if row.get("error"):
                lines.append(f"{row['episode']}: {row['error']}")
            continue
        change = row["selected_reference_change"]
        if change["available"]:
            parts = [f"{side} wrist mean/max {number(hand['wrist_translation_mean_cm'])}/{number(hand['wrist_translation_max_cm'])} cm, finger mean/max {number(hand['finger_joint_abs_mean_deg'])}/{number(hand['finger_joint_abs_max_deg'])} deg"
                     for side, hand in change["hands"].items()]
            lines.append(f"{row['episode']}: selected commands changed {change['changed_frames']}/{change['frames']} frames; " + "; ".join(parts) + ".")
        if row["search"].get("stopped"):
            lines.append(f"{row['episode']}: search stopped: {row['search']['stopped']}; MP-SR still requires a complete final rollout.")
        for issue in row.get("issues", []):
            lines.append(f"{row['episode']}: {issue}")
    if total["all_requested_complete"]:
        lines.append(f"\nAll-requested MP-SR: {number(total['baseline']['full_requested_mp_sr'])} → {number(total['refined']['full_requested_mp_sr'])}; physics-gated: {number(total['baseline']['full_requested_physical_mp_sr'])} → {number(total['refined']['full_requested_physical_mp_sr'])}.")
    else:
        lines.append("\nFull requested-set MP-SR is pending until every requested episode has complete metrics.")
    lines += ["", "Interpretation: the search objective includes contact-force and penetration penalties; reducing it does not necessarily improve MP-SR. These results cover the floating-Sharpa MPC module port with reference poses as proxy ground truth.",
              "Scene limitations: the finite tabletop has no room floor beneath it, so a dropped object's continued fall can inflate distance errors. Episode 1 also retains the supplied pot/lid mesh overlap. See the [experiment documentation](../README.md) for source assumptions, drop audits and comparison videos."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(DEFAULT_EPISODES))
    parser.add_argument("--output-prefix", type=Path, help="Write PREFIX.json and PREFIX.md; defaults to reports/TAG_seedSEED")
    args = parser.parse_args()
    report = summarize(args.workspace, args.tag, args.seed, args.episodes)
    prefix = args.output_prefix or args.workspace / "reports" / f"{args.tag}_seed{args.seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = render(report)
    prefix.with_suffix(".md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
