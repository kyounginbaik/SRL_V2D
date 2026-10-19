#!/usr/bin/env python3
"""Fresh, unassisted episode-37 policy rollout, including causal robot actions.

Run inside the source run's pinned FlashCHORD image with its code at /workspace.
The selected world is fixed in advance (28 of 32), matching epi37.json. This
script never selects the best fresh world or modifies the source run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


POLICY_SHA256 = "46653141bf7c97a95e9dcc9a2edc8be0c997ca514ecc1fec1ff4aef9f2e9ab35"
REFINED_REFERENCE_SHA256 = "97e060cd018db90724a1746e09e3664f77a0b03b8ba0fba02f7bb101d39b73b3"
TRAINING_REL = Path("runs/pilot/episode_000037/chunk/seed_44/training")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def hash_sources(root):
    """Bind all runtime Python/config sources, including this actual exporter."""
    entries = []
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
                entries.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    entries.append({"path": "exporter/export_ilchold_rollout.py", "sha256": sha256(__file__)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return entries, hashlib.sha256(payload).hexdigest()


def remap_strings(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [remap_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: remap_strings(item, old, new) for key, item in value.items()}
    return value


def prepare_inputs(root, output, old_root):
    """Copy assets; relocate path strings only, never modify reference poses."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = root / "frozen/episode_000037"
    copied = output / "runtime_inputs/episode_000037"
    shutil.copytree(source, copied)
    table = pq.read_table(source / "reference.parquet")
    rows = table.to_pylist()
    relocated = remap_strings(rows, old_root + "/frozen/episode_000037", str(copied))
    relocated = remap_strings(relocated, str(source), str(copied))
    relocated = remap_strings(relocated, old_root, str(root))
    rewritten = pa.Table.from_pylist(relocated, schema=table.schema)
    for name in table.column_names:
        if name not in ("object_mesh_paths", "object_urdf_paths"):
            if not table[name].equals(rewritten[name]):
                raise ValueError(f"Relocation would alter non-asset reference field: {name}")
    pq.write_table(rewritten, copied / "reference.parquet")
    assert np.array_equal(np.asarray(rows[0]["object_body_position"]),
                          np.asarray(relocated[0]["object_body_position"]))
    if (root / "asset_cache").exists():
        shutil.copytree(root / "asset_cache", output / "_cache/assets", dirs_exist_ok=True)
    if (root / "warp_cache").exists():
        shutil.copytree(root / "warp_cache", output / "_cache/warp", dirs_exist_ok=True)
    return copied


def metric_cohort(achieved, reference, *, first_frame=0):
    """Strict (no symmetry relaxation) first-object alignment against source proxy."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    actual = np.asarray(achieved, dtype=np.float64)[first_frame:]
    target = np.asarray(reference, dtype=np.float64)[first_frame:]
    position, orientation = [], []
    for world in range(actual.shape[1]):
        alignment = Rotation.from_quat(target[0, 0, 3:]) * Rotation.from_quat(actual[0, world, 0, 3:]).inv()
        offset = target[0, 0, :3] - alignment.apply(actual[0, world, 0, :3])
        p, q = [], []
        for obj in range(actual.shape[2]):
            pose = actual[:, world, obj]
            p.append(float(np.linalg.norm(alignment.apply(pose[:, :3]) + offset - target[:, obj, :3], axis=-1).mean()))
            q.append(float(np.degrees(((alignment * Rotation.from_quat(pose[:, 3:])) *
                                      Rotation.from_quat(target[:, obj, 3:]).inv()).magnitude()).mean()))
        position.append(p)
        orientation.append(q)
    p, q = np.asarray(position), np.asarray(orientation)
    return {"alignment": f"one rigid transform at recorded frame {first_frame}, object slot 0",
            "first_included_frame": first_frame, "frame_count": len(target),
            "position_mean_m_per_world_object": p.tolist(),
            "orientation_mean_deg_per_world_object": q.tolist(),
            "strict_mp_sr_per_world": ((p < .03) & (q < 30.)).all(axis=1).tolist(),
            "strict_mp_sr_world_fraction": float(((p < .03) & (q < 30.)).all(axis=1).mean()),
            "reference_is_reconstruction_proxy_not_hidden_ground_truth": True,
            "orientation_symmetry": "identity only (stricter than symmetry-aware scoring)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--world-count", type=int, default=32)
    parser.add_argument("--world-index", type=int, default=28)
    args = parser.parse_args()
    root, output, runtime = (p.expanduser().resolve() for p in
                             (args.run_root, args.output, args.runtime_root))
    if output.exists():
        raise FileExistsError(f"Require a new output directory: {output}")
    if root in output.parents or output == root:
        raise ValueError("Output must be outside the immutable source run")
    if args.world_count != 32 or args.world_index != 28:
        raise ValueError("This video-matched export fixes cohort=32 and selected world=28")
    policy = root / TRAINING_REL / "policy_2508800.safetensors"
    refined = root / TRAINING_REL / "references/reference_0005.npz"
    source_reference = root / "frozen/episode_000037/reference.parquet"
    demo = json.loads((root / "demo/epi37.json").read_text())
    if sha256(policy) != POLICY_SHA256 or demo["policy_sha256"] != POLICY_SHA256:
        raise ValueError("Selected checkpoint differs from the user's epi37.mp4")
    if sha256(refined) != REFINED_REFERENCE_SHA256:
        raise ValueError("Selected refined reference differs from the user's epi37.mp4")
    if Path(demo["policy"]).relative_to(demo["root"]) != policy.relative_to(root):
        raise ValueError("Video metadata names a different policy checkpoint")
    if Path(demo["final_reference"]).relative_to(demo["root"]) != refined.relative_to(root):
        raise ValueError("Video metadata names a different refined reference")
    if demo["refined_world"] != args.world_index or demo["frames"] != 151:
        raise ValueError("Unexpected original video world or timeline")
    output.mkdir(parents=True)
    os.environ["FLASH_CHORD_CACHE_DIR"] = str(output / "_cache/assets")
    os.environ["XDG_CACHE_HOME"] = str(output / "_cache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime / "src"))

    import jax
    import jax.numpy as jnp
    import numpy as np
    import warp as wp
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import apply_evaluation_composition
    from flash_chord.evaluation.submission import write_object_trajectories_parquet
    from flash_chord.runtime.jax import to_jax
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

    wp.config.kernel_cache_dir = str(output / "_cache/warp")
    config_path = root / TRAINING_REL.parent / "run_config.json"
    launch = json.loads(config_path.read_text())
    old_root = demo["root"]
    frozen = prepare_inputs(root, output, old_root)
    removed = ("scripts/", "output_dir=", "hydra.", "+bilevel", "+adaptive_voc", "logging.", "task.robot_reference")
    overrides = [item.replace(old_root, str(root)) for item in launch["training_command"]
                 if not item.startswith(removed)]
    overrides = [item for item in overrides if not item.startswith("task.parquet=")]
    overrides.append(f"task.parquet={frozen / 'reference.parquet'}")
    with initialize_config_dir(config_dir=str(runtime / "src/flash_chord/configs"), version_base="1.3"):
        cfg = compose(config_name="train_flash_sac", overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    apply_evaluation_composition(resolved, world_count=args.world_count)
    env_cfg = instantiate_typed(OmegaConf.create(resolved["env"]), RLEnvConfig)
    training = instantiate_typed(cfg.training, TrainingConfig)
    if env_cfg.voc_scale != 0. or env_cfg.voc.max_force != 0. or env_cfg.voc.max_torque != 0. or env_cfg.auto_reset:
        raise ValueError("Evaluation must have zero object assistance and no automatic resets")
    dump(output / "resolved_evaluation_config.json", resolved)
    sources, source_code_sha = hash_sources(runtime)
    dump(output / "source_code_manifest.json", {"files": sources, "sha256": source_code_sha})

    def make_setup(ref):
        kw = {} if ref is None else {"robot_reference_path": str(ref), "robot_reference_sha256": sha256(ref)}
        return setup_scene(parquet=str(frozen / "reference.parquet"), control_fps=env_cfg.sim.fps,
                           motion_speed=1.0, embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                           collision=instantiate_typed(cfg.collision, CollisionPolicy),
                           world_count=args.world_count, include_support=resolved["scene"]["support"],
                           decompose_objects=True, **kw)

    world, worlds = args.world_index, args.world_count
    with wp.ScopedDevice("cuda:0"):
        setup = make_setup(refined)
        # The original renderer builds this second scene before its refined rollout.
        ik_setup = make_setup(None)
        env = RLEnv(setup.scene, setup.reference, config=env_cfg)
        env.reset(frame_id=0)
        env.capture_step()
        env.reset(frame_id=0)
        adapter = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
        obs = jnp.array(to_jax(env.reset(frame_id=0), (worlds, env.observation_dim)), copy=True)
        actor_state, actor, key, _ = load_policy_for_inference(str(policy), training, obs,
                                                              adapter.action_dim, deterministic=True)
        scene = setup.scene
        count = scene.robot_reference.num_frames
        fps = float(scene.robot_reference.fps)
        if count != 151 or fps != 20.0 or adapter.action_dim != 56:
            raise ValueError(f"Unexpected video contract: {count=}, {fps=}, {adapter.action_dim=}")
        body_count = scene.model.body_count // worlds
        ids = env.command.body_ids_w.numpy().reshape(worlds, -1)
        local_ids = ids[world] - world * body_count
        names = setup.reference.object_body_names()
        if len(names) != 1 or len(ids[world]) != 1:
            raise ValueError("This exporter expects episode37's single rigid spoon")
        target = np.concatenate((np.asarray(setup.reference.object_body_pos_w()),
                                 np.asarray(setup.reference.object_body_quat_w())[..., [1, 2, 3, 0]]), axis=-1)
        # Do not read reset command target buffers: they are intentionally empty until step1.
        if not np.allclose(np.linalg.norm(target[..., 3:], axis=-1), 1., atol=1e-5):
            raise ValueError("Invalid source target quaternion")

        def slice_world(array):
            return np.asarray(array.numpy()).reshape(worlds, -1).copy()[world]

        states = {name: [] for name in ("body_q", "body_qd", "joint_q", "joint_qd", "observation", "timestep")}
        controls = {name: [] for name in ("policy_action", "mapped_action", "filtered_residual",
                                         "processed_target", "joint_target", "joint_target_pos",
                                         "joint_force_last_substep", "command_reference_frame", "terminated", "truncated")}
        all_objects = []
        voc_scales = []

        def capture_state():
            body = env.state_0.body_q.numpy().copy()
            states["body_q"].append(body.reshape(worlds, body_count, 7)[world])
            states["body_qd"].append(env.state_0.body_qd.numpy().reshape(worlds, body_count, -1)[world].copy())
            for name in ("joint_q", "joint_qd"):
                states[name].append(slice_world(getattr(env.state_0, name)))
            states["observation"].append(np.asarray(jax.device_get(obs))[world].copy())
            states["timestep"].append(int(env.timestep.numpy()[world]))
            all_objects.append(body[ids].copy())
            voc_scales.append(env.object_control_scale.numpy().copy())

        reset_state = {"sequence": ["reset(frame_id=0)", "capture_step()", "reset(frame_id=0)",
                                    "construct adapter", "reset(frame_id=0)", "load deterministic policy"],
                       "config": asdict(env_cfg.reset) if is_dataclass(env_cfg.reset) else str(env_cfg.reset),
                       "evaluation_seed": resolved["env"]["reset"]["seed"],
                       "training_seed": launch["seed"], "world_count": worlds, "world_index": world,
                       "settling_steps": 0, "first_record_is_reset_state": True}
        for name in ("reset_count", "finger_scale", "steps_since_reset", "applied_voc_scale", "reset_frame"):
            if hasattr(env.reset_policy, name):
                reset_state[name] = getattr(env.reset_policy, name).numpy().tolist()
        capture_state()
        for frame in range(count - 1):
            controls["command_reference_frame"].append(int(env.timestep.numpy()[world]))
            action, key = actor(actor_state, obs, key)
            controls["policy_action"].append(np.asarray(jax.device_get(action))[world].copy())
            transition = adapter.step(action)
            obs = jnp.array(transition.observation, copy=True)
            controls["mapped_action"].append(slice_world(env.action.raw_action))
            controls["filtered_residual"].append(slice_world(env.action.filtered_action))
            controls["processed_target"].append(slice_world(env.action.processed_target))
            controls["joint_target"].append(slice_world(env.action.joint_target))
            controls["joint_target_pos"].append(slice_world(env.control.joint_target_pos))
            controls["joint_force_last_substep"].append(slice_world(env.control.joint_f))
            controls["terminated"].append(int(env.done.numpy()[world]))
            controls["truncated"].append(int(env.truncation.numpy()[world]))
            capture_state()
            if (frame + 1) % 25 == 0:
                print(f"Recorded {frame+1}/{count-1} physical control transitions", flush=True)
        arrays = {name: np.asarray(values) for name, values in {**states, **controls}.items()}
        all_objects = np.stack(all_objects)
        if not all(np.isfinite(value).all() for value in arrays.values()) or not np.isfinite(all_objects).all():
            raise ValueError("Non-finite physical rollout")
        if np.any(np.asarray(voc_scales) != 0):
            raise ValueError("Object assistance became nonzero during rollout")
        arrays.update(time_s=np.arange(count, dtype=np.float64) / fps,
                      action_time_s=np.arange(count-1, dtype=np.float64) / fps,
                      achieved_object_pose_xyzw=all_objects[:, world], reference_object_pose_xyzw=target,
                      object_body_local_ids=local_ids, object_body_names=np.asarray(names),
                      robot_joint_q=arrays["joint_q"][:, :scene.layout.num_joint_q],
                      robot_joint_qd=arrays["joint_qd"][:, :scene.layout.num_joint_dof],
                      robot_joint_target=arrays["joint_target"][:, :scene.layout.num_joint_dof],
                      robot_reference_joint_q=np.asarray(scene.robot_reference.joint_q),
                      object_assistance_scale=np.asarray(voc_scales)[:, world])
        np.savez_compressed(output / "actions.npz", **arrays)
        np.savez_compressed(output / "all_world_object_poses.npz", achieved_pose_xyzw=all_objects,
                            reference_pose_xyzw=target, selected_world_index=world, fps=fps)
        dump(output / "reset_metadata.json", reset_state)
        action_layout = env.action.buffers.layout
        joints_per_world = scene.model.joint_count // worlds
        joint_slice = slice(world*joints_per_world, (world+1)*joints_per_world+1)
        layout = {"action": asdict(action_layout), "body_labels_selected_world":
                  [str(x) for x in scene.model.body_label[world*body_count:(world+1)*body_count]],
                  "joint_labels_selected_world": [str(x) for x in scene.model.joint_label[
                      world*(scene.model.joint_count//worlds):(world+1)*(scene.model.joint_count//worlds)]],
                  "joint_q_start_selected_world": (scene.model.joint_q_start.numpy()[joint_slice]
                      - world*(scene.model.joint_coord_count//worlds)).tolist(),
                  "joint_qd_start_selected_world": (scene.model.joint_qd_start.numpy()[joint_slice]
                      - world*(scene.model.joint_dof_count//worlds)).tolist(),
                  "joint_start_semantics": "local coordinate offsets; final entry is end sentinel"}
        dump(output / "robot_layout.json", layout)

    recorded = np.load(root / "demo/epi37_parts/rollouts.npz", allow_pickle=False)
    previous = recorded["refined"]
    current = arrays["body_q"]
    if previous.shape != current.shape or int(recorded["world_refined"]) != world:
        raise ValueError("Original demo recording layout/world mismatch")
    delta_pos = np.linalg.norm(current[..., :3] - previous[..., :3], axis=-1)
    delta_rot = np.degrees((Rotation.from_quat(current[..., 3:].reshape(-1, 4)) *
                            Rotation.from_quat(previous[..., 3:].reshape(-1, 4)).inv()).magnitude())
    comparison = {"source_sha256": sha256(root / "demo/epi37_parts/rollouts.npz"),
                  "bitwise_identical_body_q": bool(np.array_equal(current, previous)),
                  "max_body_position_difference_m": float(delta_pos.max()),
                  "max_body_orientation_difference_deg": float(delta_rot.max()),
                  "mean_body_position_difference_m": float(delta_pos.mean()),
                  "first_frame_max_abs_difference": float(np.abs(current[0]-previous[0]).max())}
    comparison["numerically_matches_original_demo"] = bool(delta_pos.max() <= 1e-5 and delta_rot.max() <= 1e-3)
    comparison["numerical_match_tolerances"] = {"max_body_position_m": 1e-5, "max_body_orientation_deg": 1e-3}
    comparison["video_binding"] = ("bitwise_exact" if comparison["bitwise_identical_body_q"] else
        "numerically_matches_with_declared_tolerances" if comparison["numerically_matches_original_demo"] else
        "different_fresh_rollout_requires_its_own_video")
    strict = metric_cohort(all_objects, target)
    historical = metric_cohort(all_objects, target, first_frame=1)
    rollout = SimpleNamespace(achieved_pose_w=all_objects[:, world:world+1], reference_pose_w=target,
                              object_body_names=names, body_object_ids=np.zeros(len(names), np.int32),
                              non_finite_worlds=(), start_frame=0, control_fps=fps)
    write_object_trajectories_parquet(output / "episode_000037.parquet", rollout,
                                     checkpoint_sha256=sha256(policy), eval_script_sha256=sha256(__file__),
                                     eval_code_sha256=source_code_sha, reference_parquet=source_reference)
    packages = {}
    for name in ("jax", "jaxlib", "warp-lang", "newton", "mujoco", "mujoco-warp", "numpy", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed as distribution"
    report = {"schema": "uniretarget_fresh_ilchold_rollout_v1", "episode": "episode_000037",
              "checkpoint_sha256": sha256(policy), "checkpoint_relative_path": str(policy.relative_to(root)),
              "source_reference_sha256": sha256(source_reference), "reference_refinement_sha256": sha256(refined),
              "run_config_sha256": sha256(config_path), "source_run_root": str(root),
              "selected_world_index": world, "world_count": worlds, "world_selection": "fixed from supplied video's metadata before fresh evaluation",
              "frame_count": count, "frames": count, "world_index": world,
              "fps": fps, "physics_substeps_per_control_step": env_cfg.sim.substeps,
              "physics_fps": fps * env_cfg.sim.substeps, "policy_action_dimension": adapter.action_dim,
              "physical_transitions": count-1, "no_object_assistance": True, "voc_scale": 0.,
              "voc_max_force": 0., "voc_max_torque": 0., "automatic_reset": False,
              "finite_states_and_actions": True, "object_assistance_enabled": False,
              "autoreset_enabled": False, "policy_actions_recorded": True,
              "strict_reference_proxy_mp_sr": float(strict["strict_mp_sr_per_world"][world]),
              "refined_reference_sha256": sha256(refined),
              "strict_source_reference_metrics_all_151_frames": strict,
              "historical_demo_comparison_frames_1_through_150": historical,
              "selected_world_strict_proxy_pass": strict["strict_mp_sr_per_world"][world],
              "original_demo_comparison": comparison,
              "source_code_sha256": source_code_sha, "eval_script_sha256": sha256(__file__),
              "packages": packages, "jax_devices": [str(device) for device in jax.devices()],
              "actions_semantics": {
                  "causal_pair": "observation[k], policy_action[k] -> state[k+1]; state[0] is reset; 151 states and150 actions",
                  "policy_action": "56 dimension raw actor residual input, right hand then left; 3 translation+3 residual Euler rotation+22finger coordinates per hand; not direct servo targets",
                  "filtered_residual": "mapped, scaled, clipped, EMA-smoothed residual in m/rad applied to refined reference",
                  "processed_target": "each hand: xyz meters, quaternion wxyz, 22 finger radians; right then left",
                  "joint_target": "simulation DOF ordered target; wrist orientation is world rotation-vector, finger angles radians; includes passive object slots",
                  "joint_target_pos": "actual Newton position-drive targets used during transition; wrist effort controller acts separately",
                  "joint_force_last_substep": "last physics-substep controller generalized force snapshot only; not every substep force history",
                  "observation_and_state": "state and observation sampled at20Hz; internal simulation400Hz; no fabricated500Hz action samples",
                  "time_alignment": "physical state[k] at k/20 s compared to original source reference frame k; command_reference_frame[k] records frame actually targeted in transition k",
                  "body_quaternion_order": "xyzw", "object_reference_quaternion_order": "xyzw"}}
    dump(output / "report.json", report)
    inventory = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                 if path.is_file() and "_cache" not in path.relative_to(output).parts}
    dump(output / "SHA256SUMS.json", inventory)
    print(json.dumps({"output": str(output), "selected_world_strict_proxy_pass": report["selected_world_strict_proxy_pass"],
                      "original_demo_comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
