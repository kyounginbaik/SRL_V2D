"""Tissue-box UniRetargeting preflight: visual-geometry proposals, physical acceptance.

This is a frozen-lower-policy upper step, not an end-to-end training result.
It tests whether a bounded robot reference can reduce finger/box overlap while
preserving object tracking under the same FlashSAC controller and reset seed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import newton
import numpy as np
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.retargeting.penetration import VisualHullPenetration, penetration_summary
from flash_chord.retargeting.reference_snapshot import save_snapshot
from flash_chord.retargeting.ttsa_editor import (
    TTSAReferenceEditor, choose_local_centers, passes_study_gate,
    project_reference, study_basis, upper_cost,
)
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--search-stride", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--amplitude-rad", type=float, default=0.08)
    parser.add_argument("--blend", type=float, default=0.25,
                        help="fraction of the new upper action published by the online TTSA editor")
    parser.add_argument("--online-action", action="store_true",
                        help="test the exact bounded/blended action used by the TTSA editor")
    args = parser.parse_args()
    if args.worlds < 1 or args.search_stride < 1 or args.rounds < 1 or not 0 < args.amplitude_rad <= 0.12:
        parser.error("worlds/stride/rounds must be positive and amplitude in (0, 0.12]")
    args.output.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(read_checkpoint_metadata(args.checkpoint))
    cfg = OmegaConf.create(config)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet, control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed, embodiment=embodiment, collision=collision,
        world_count=args.worlds, include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        object_initial_pose_overrides=cfg.scene.get("object_initial_pose_overrides"),
        source_frame_playback=cfg.task.source_frame_playback,
        motion_start_frame=cfg.task.motion_start_frame,
        motion_end_frame=cfg.task.motion_end_frame,
        robot_reference_path=cfg.task.get("robot_reference_path"),
        robot_reference_sha256=cfg.task.get("robot_reference_sha256"),
    )
    editor = TTSAReferenceEditor(
        options=dict(mode="reactor_wrench", engine="ttsa_flashsac", study_variant="global_local",
                     penetration_aware=True, penetration_measure="visual_hull",
                     guided_penetration_proposal=args.online_action,
                     select_best_trial=args.online_action,
                     penetration_reference_stride=8, penetration_rollout_stride=8,
                     eval_worlds=args.worlds, upper_trials=4, interval=1, warmup=1,
                     blend=args.blend),
        setup=setup, embodiment=embodiment, collision=collision,
        training=training, environment_config=deepcopy(config),
        checkpoint_metadata={}, output_dir=args.output,
    )
    env = RLEnv(editor.eval_setup.scene, editor.eval_setup.reference, config=deepcopy(editor.eval_config))
    env.reset(frame_id=0)
    adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    observation = jnp.array(adapter.reset(), copy=True)
    actor, _, _, _ = load_policy_for_inference(
        str(args.checkpoint), training, observation, adapter.action_dim, deterministic=True,
    )
    lower = SimpleNamespace(actor=actor)
    baseline, _, _, feedback = editor.rollout(editor.binding, lower, args.seed, return_feedback=True)

    if args.online_action:
        editor.upper_round = 1
        action, search = editor.guided_action()
        candidate_binding = editor.make_binding(action) if action is not None else editor.binding
        candidate, _, _ = editor.rollout(candidate_binding, lower, args.seed)
        gated = passes_study_gate(candidate, baseline, penetration_aware=True)
        base_cost = upper_cost(baseline, np.zeros_like(editor.baseline.joint_q),
                               editor.baseline.fps, 1.0, robust=True, penetration_weight=1.0)
        candidate_cost = upper_cost(candidate, candidate_binding.joint_q-editor.baseline.joint_q,
                                    editor.baseline.fps, 1.0, robust=True, penetration_weight=1.0)
        accepted = bool(action is not None and gated and candidate_cost < base_cost)
        if accepted:
            save_snapshot(args.output / "accepted_reference.npz", candidate_binding, editor.setup.scene.layout,
                          provenance={"method": "uniretargeting_online_action_frozen_policy", "seed": args.seed,
                                      "source_checkpoint": str(args.checkpoint)})
        report = dict(checkpoint=str(args.checkpoint), seed=args.seed, worlds=args.worlds,
                      lower_policy_frozen=True, proposal="exact_online_ttsa_guided_action",
                      blend=args.blend,
                      search=search, baseline=baseline, candidate=candidate,
                      baseline_cost=base_cost, candidate_cost=candidate_cost,
                      passes_gate=bool(gated), accepted=accepted,
                      max_joint_change_rad=float(np.max(np.abs(candidate_binding.joint_q-editor.baseline.joint_q))),
                      max_added_velocity_rad_s=float(np.max(np.abs(np.diff(
                          candidate_binding.joint_q-editor.baseline.joint_q,axis=0))) * editor.baseline.fps))
        (args.output / "guided_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("search", "baseline_cost", "candidate_cost",
                                                       "passes_gate", "accepted")}, indent=2), flush=True)
        return

    monitor = VisualHullPenetration(env.scene, samples_per_shape=8)
    frames = np.unique(np.r_[np.arange(0, editor.baseline.num_frames, args.search_stride),
                              editor.baseline.num_frames - 1])
    q_per_world = env.model.joint_coord_count // env.world_count
    robot_q = editor.baseline.num_joint_q

    def visual_measure(q):
        depths = np.empty((len(frames), env.world_count, len(monitor.sides)))
        for slot, frame in enumerate(frames):
            env.reset(frame_id=int(frame))
            state_q = env.state_0.joint_q.numpy()
            for world in range(env.world_count):
                state_q[world * q_per_world:world * q_per_world + robot_q] = q[frame]
            env.state_0.joint_q.assign(np.asarray(state_q, dtype=np.float32))
            newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
            depths[slot] = monitor.frame_depths(env.state_0)
        return penetration_summary(depths)

    baseline_visual = visual_measure(editor.baseline.joint_q)
    centers = choose_local_centers(feedback["penetration_frame_scores"], count=4)
    if len(centers) < 4:
        centers = [.2, .4, .6, .8]
    basis = study_basis(editor.baseline.num_frames, "global_local", local_centers=centers)
    modes = (0, 3, 4, 5, 6)
    fingers = tuple(i for i in range(len(editor.joint_basis)) if i % 12 >= 7)
    current = editor.baseline.joint_q.copy()
    current_visual = baseline_visual
    history = []

    def visual_score(metrics):
        return metrics["penetration_mean_m"] / .005 + .5 * metrics["penetration_peak_m"] / .05

    for round_id in range(args.rounds):
        best = (visual_score(current_visual), current, current_visual, None)
        evaluated = 0
        for mode in modes:
            for finger in fingers:
                for sign in (-1.0, 1.0):
                    delta = (current - editor.baseline.joint_q) + (
                        sign * args.amplitude_rad * basis[:, mode, None] * editor.joint_basis[finger]
                    )
                    try:
                        candidate = project_reference(
                            editor.baseline.joint_q, delta, editor.projector.qids,
                            editor.projector.lower, editor.projector.upper, editor.baseline.fps,
                        )
                    except ValueError:
                        continue
                    measure = visual_measure(candidate)
                    evaluated += 1
                    # Never trade a much deeper single frame for a lower mean.
                    if measure["penetration_peak_m"] > baseline_visual["penetration_peak_m"] + .002:
                        continue
                    score = visual_score(measure)
                    if score < best[0] - 1e-4:
                        best = (score, candidate, measure, dict(mode=mode, finger=finger, sign=sign))
        if best[3] is None:
            history.append(dict(round=round_id, evaluated=evaluated, accepted=False))
            break
        current, current_visual = best[1], best[2]
        history.append(dict(round=round_id, evaluated=evaluated, accepted=True,
                            choice=best[3], visual_metrics=current_visual))
        print(f"guided round {round_id+1}: {best[3]} visual mean={current_visual['penetration_mean_m']:.4f}m",
              flush=True)

    target = editor.baseline.joint_target.copy()
    target[:, editor.projector.dids] = current[:, editor.projector.qids]
    candidate_binding = build_robot_reference_binding(
        editor.projector.builder, editor.projector.layout, current, target, editor.baseline.fps,
    )
    candidate, _, _ = editor.rollout(candidate_binding, lower, args.seed)
    gated = passes_study_gate(candidate, baseline, penetration_aware=True)
    base_cost = upper_cost(baseline, np.zeros_like(current), editor.baseline.fps, 1.0,
                           robust=True, penetration_weight=1.0)
    candidate_cost = upper_cost(candidate, current - editor.baseline.joint_q,
                                editor.baseline.fps, 1.0, robust=True, penetration_weight=1.0)
    accepted = bool(gated and candidate_cost < base_cost)
    if accepted:
        save_snapshot(args.output / "accepted_reference.npz", candidate_binding, editor.setup.scene.layout,
                      provenance={"method": "uniretargeting_guided_frozen_policy", "seed": args.seed,
                                  "source_checkpoint": str(args.checkpoint)})
    report = dict(
        checkpoint=str(args.checkpoint), seed=args.seed, worlds=args.worlds,
        lower_policy_frozen=True, geometry="visible finger samples vs convex hull of visible box mesh",
        baseline=baseline, candidate=candidate, baseline_visual_search=baseline_visual,
        candidate_visual_search=current_visual, history=history,
        baseline_cost=base_cost, candidate_cost=candidate_cost,
        passes_gate=bool(gated), accepted=accepted,
        max_joint_change_rad=float(np.max(np.abs(current-editor.baseline.joint_q))),
        max_added_velocity_rad_s=float(np.max(np.abs(np.diff(current-editor.baseline.joint_q, axis=0))) * editor.baseline.fps),
    )
    (args.output / "guided_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("baseline_visual_search", "candidate_visual_search",
                                                   "baseline_cost", "candidate_cost", "passes_gate", "accepted")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
