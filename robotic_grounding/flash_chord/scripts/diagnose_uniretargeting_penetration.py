"""Measure collision-shape finger/object penetration in a kinematic reference.

Uses Newton's collision query at each posed reference frame; it does not step
the simulator or infer penetration from rendered meshes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.retargeting.penetration import HandObjectPenetration, VisualHullPenetration, penetration_summary
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--visual-proxy", action="store_true",
                        help="also sample visible hand meshes against the box's convex visual hull")
    parser.add_argument("--samples-per-shape", type=int, default=12)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    config = checkpoint_config(read_checkpoint_metadata(args.checkpoint))
    cfg = OmegaConf.create(config)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet, control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed, embodiment=embodiment, collision=collision,
        world_count=1, include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        object_initial_pose_overrides=cfg.scene.get("object_initial_pose_overrides"),
        source_frame_playback=cfg.task.source_frame_playback,
        motion_start_frame=cfg.task.motion_start_frame,
        motion_end_frame=cfg.task.motion_end_frame,
        robot_reference_path=str(args.reference),
    )
    env = RLEnv(setup.scene, setup.reference, config=env_config)
    monitor = HandObjectPenetration(setup.scene)
    visual = VisualHullPenetration(setup.scene, samples_per_shape=args.samples_per_shape) if args.visual_proxy else None
    frames = np.arange(0, setup.scene.robot_reference.num_frames, args.stride, dtype=int)
    depths = np.zeros((len(frames), 1, len(monitor.sides)), dtype=np.float64)
    selected_counts = np.zeros(len(frames), dtype=int)
    minimum_gaps = np.full(len(frames), np.nan, dtype=float)
    visual_depths = np.zeros_like(depths) if visual is not None else None
    active = np.stack([
        np.asarray(setup.reference.contact_active(side)).reshape(setup.reference.num_frames, -1).any(axis=1)
        for side in monitor.sides
    ], axis=1)
    for i, frame in enumerate(frames):
        env.reset(frame_id=int(frame))
        contacts = monitor.collide(env.state_0)
        selected = monitor.selected_gaps(env.state_0, contacts)
        selected_counts[i] = len(selected)
        if selected:
            minimum_gaps[i] = min(gap for _, _, gap in selected)
        depths[i] = monitor.frame_depths(env.state_0, contacts, active[int(frame)])
        if visual is not None:
            visual_depths[i] = visual.frame_depths(env.state_0)
        if (i + 1) % 100 == 0:
            print(f"measured {i+1}/{len(frames)} reference frames", flush=True)
    report = {
        "checkpoint": str(args.checkpoint), "reference": str(args.reference),
        "sides": monitor.sides, "frames": frames.tolist(), "stride": args.stride,
        "metrics": penetration_summary(depths),
        "by_side": {side: penetration_summary(depths[:, :, side_id:side_id+1])
                    for side_id, side in enumerate(monitor.sides)},
        "depth_by_frame_side_m": depths[:, 0].tolist(),
        "selected_hand_object_contact_count": int(selected_counts.sum()),
        "frames_with_hand_object_contacts": int(np.count_nonzero(selected_counts)),
        "minimum_selected_gap_m": float(np.nanmin(minimum_gaps)) if np.any(selected_counts) else None,
        "interpretation": "Newton collision-shape overlap; 2 mm tolerated on labeled-contact frames",
    }
    if visual is not None:
        report["visual_hull_proxy"] = {
            "metrics": penetration_summary(visual_depths),
            "sample_count_by_side": dict(zip(visual.sides, visual.sample_count_by_side, strict=True)),
            "depth_by_frame_side_m": visual_depths[:, 0].tolist(),
            "caveat": "Convex hull fills tissue mesh holes; sampled visual overlap is not Newton collision penetration",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"],
                      "selected_hand_object_contact_count": report["selected_hand_object_contact_count"],
                      "minimum_selected_gap_m": report["minimum_selected_gap_m"],
                      "visual_hull_proxy": report.get("visual_hull_proxy", {}).get("metrics")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
