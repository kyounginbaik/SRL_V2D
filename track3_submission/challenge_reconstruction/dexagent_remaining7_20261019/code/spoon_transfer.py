#!/usr/bin/env python3
"""Geometry-registered measured spoon grasp proposals; robot commands only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from prepare_seven import BASE, ROOT, LEGACY, sha, write
sys.path.insert(0, str(LEGACY / "code"))
sys.path.insert(0, str(ROOT / "third_party/DexAgent"))
from tool_library.common import hand as H, tracking as TR
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from physics_pickup_search_v2 import PickupExperiment

DONOR = LEGACY / "refinement_20261018/episode_000003/retention_after45"
OUT = BASE / "diagnostics/spoon_transfer"


def rigid_alignment(source_mesh, target_mesh, output):
    """Register object surfaces, never reference trajectories or achieved poses."""
    source = trimesh.load(source_mesh, force="mesh", process=False)
    target = trimesh.load(target_mesh, force="mesh", process=False)
    a, _ = trimesh.sample.sample_surface(source, 10000, seed=413)
    b, _ = trimesh.sample.sample_surface(target, 14000, seed=414)
    tree = cKDTree(b)
    ca, cb = a.mean(0), b.mean(0)
    _, va = np.linalg.eigh(np.cov(a.T)); _, vb = np.linalg.eigh(np.cov(b.T))
    initial = [np.eye(3)]
    import itertools
    for signs in itertools.product([-1., 1.], repeat=3):
        matrix = vb @ np.diag(signs) @ va.T
        if np.linalg.det(matrix) > 0:
            initial.append(matrix)
    candidates = []
    for rotation in initial:
        translation = cb - rotation @ ca
        for _ in range(50):
            transformed = a @ rotation.T + translation
            distances, indices = tree.query(transformed)
            keep = distances <= np.quantile(distances, .90)
            x, y = a[keep], b[indices[keep]]
            x0, y0 = x.mean(0), y.mean(0)
            u, _, vt = np.linalg.svd((x - x0).T @ (y - y0))
            correction = np.eye(3); correction[2, 2] = np.linalg.det(vt.T @ u.T)
            rotation = vt.T @ correction @ u.T
            translation = y0 - rotation @ x0
        transformed = a @ rotation.T + translation
        forward = tree.query(transformed)[0]
        reverse = cKDTree(transformed).query(b)[0]
        score = float(np.mean(np.minimum(forward, .01)) + np.mean(np.minimum(reverse, .01)))
        candidates.append(dict(rotation=rotation.tolist(), translation_m=translation.tolist(),
                               trimmed_bidirectional_score_m=score,
                               source_to_target_mean_distance_m=float(forward.mean()),
                               source_to_target_p95_distance_m=float(np.quantile(forward, .95)),
                               target_to_source_mean_distance_m=float(reverse.mean()),
                               rotation_angle_degrees=float(np.degrees(Rotation.from_matrix(rotation).magnitude()))))
    best = min(candidates, key=lambda candidate: candidate["trimmed_bidirectional_score_m"])
    if best["source_to_target_p95_distance_m"] > .008:
        raise ValueError("Donor/target geometry mismatch too large for direct grasp transfer")
    matrix = np.eye(4); matrix[:3, :3] = best["rotation"]; matrix[:3, 3] = best["translation_m"]
    report = dict(source_mesh=str(source_mesh), target_mesh=str(target_mesh), source_sha256=sha(source_mesh),
                  target_sha256=sha(target_mesh), target_body_from_donor_body=matrix.tolist(),
                  best=best, candidates=candidates, rigid_only=True, scale_adjustment=False,
                  references_used_for_registration=False,
                  interpretation="Geometric command seed; collision and mass differ, so successful donor contact is not a target success certificate")
    write(output, report)
    return matrix, report


def donor_grasp(anchor_s):
    model = mujoco.MjModel.from_xml_path(str(DONOR / "execution/scene_physics.xml"))
    record = np.load(DONOR / "execution/trajectory.npz", allow_pickle=False)
    frame = round(anchor_s * float(record["fps"]))
    data = mujoco.MjData(model); data.qpos[:] = record["qpos"][frame]; mujoco.mj_forward(model, data)
    joints = hand_joints(model, "right"); columns = model.jnt_qposadr[joints].astype(int)
    object_column = int(model.joint("wooden_spoon_free").qposadr[0])
    object_rotation = Rotation.from_quat(data.qpos[object_column + 3:object_column + 7], scalar_first=True)
    local_position = object_rotation.inv().apply(data.qpos[columns[:3]] - data.qpos[object_column:object_column + 3])
    local_rotation = object_rotation.inv() * Rotation.from_euler("XYZ", data.qpos[columns[3:6]])
    rig = H.HandRig.build(model, "robot_right_", "right")
    actuator_by_joint = {int(j): index for index, j in enumerate(model.actuator_trnid[:, 0])}
    closed = np.array([record["ctrl"][frame, actuator_by_joint[j]] for j in joints[6:]])
    object_body = model.body("wooden_spoon").id
    normals = {finger: [] for finger in ("thumb", "index")}
    for contact in data.contact:
        b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        hand = b2 if b1 == object_body else b1 if b2 == object_body else -1
        if hand < 0:
            continue
        for finger in normals:
            if f"_{finger}_" in model.body(hand).name:
                normals[finger].append(np.array(contact.frame[:3]) * (1 if b1 == object_body else -1))
    if any(not values for values in normals.values()):
        raise ValueError("Donor does not have observed thumb/index contact")
    targets = {}
    for finger, values in normals.items():
        normal = np.mean(values, axis=0); normal /= np.linalg.norm(normal)
        targets[finger] = data.site_xpos[rig.fingertip_sites[finger]].copy() + .007 * normal
    H.finger_ik(model, data, rig, targets, iters=160)
    opened = data.qpos[columns[6:]].copy()
    names = [model.joint(j).name for j in joints]
    return local_position, local_rotation, closed, opened, names


def run(episode, anchor, tag, episode_dir=None, close_shift=0., release=None, park_left=False):
    if episode not in (15, 16):
        raise ValueError("Spoon transfer diagnostic only supports episodes 15 and 16")
    name = f"episode_{episode:06d}"
    inp = Path(episode_dir) if episode_dir else BASE / "episodes" / name
    manifest = json.loads((inp / "input_manifest.json").read_text())
    directory = OUT / name / tag
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    target_mesh = Path(manifest["source"]) / "assets/wooden_spoon.obj"
    donor_manifest = json.loads((LEGACY / "episodes/episode_000003/input_manifest.json").read_text())
    donor_mesh = Path(donor_manifest["source"]) / "assets/wooden_spoon.obj"
    cached_registration = OUT / "mesh_registration_spoon.json"
    if cached_registration.exists():
        registration = json.loads(cached_registration.read_text())
        if (registration["source_sha256"] != sha(donor_mesh)
                or registration["target_sha256"] != sha(target_mesh)):
            raise ValueError("Cached spoon registration has different source/target mesh")
        transform = np.asarray(registration["target_body_from_donor_body"])
        write(directory / "mesh_registration.json", registration)
    else:
        transform, registration = rigid_alignment(donor_mesh, target_mesh, directory / "mesh_registration.json")
        write(cached_registration, registration)
    local_p, local_r, closed, opened, names = donor_grasp(anchor)
    local_p = transform[:3, :3] @ local_p + transform[:3, 3]
    local_r = Rotation.from_matrix(transform[:3, :3]) * local_r
    if episode_dir:
        from floating_runtime import _freeze
        from calibrate_controller import controller_scene
        frozen_dir = directory / "frozen_initial"; frozen_dir.mkdir()
        _, frozen = _freeze(inp / "stage2/scene.xml", frozen_dir)
        scene = directory / "controller_scene.xml"
        controller_scene(frozen, scene, "kp5000_r200_d5")
    else:
        baseline = json.loads((BASE / "runs/hawor_mpc_v1" / name / "baseline/result.json").read_text())
        scene = Path(baseline["execution"]["scene"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    joints = hand_joints(model, "right"); columns = model.jnt_qposadr[joints].astype(int)
    if names != [model.joint(j).name for j in joints]:
        raise ValueError("Donor hand joint order differs")
    original = TR.load_reference(inp / "tracking_reference.npz", model)
    q = original["qpos"]; fps = float(original["fps"])
    seed_reference = inp / "tracking_reference.npz"
    if park_left:
        left = model.jnt_qposadr[hand_joints(model, "left")].astype(int)
        parked = q.copy(); parked[:, left] = q[0, left]
        seed_reference = TR.save_reference(directory / "left_parked_seed.npz", model, parked,
            original["labels"], fps, original["segment"], original["segments"],
            source="Robot-only diagnostic: left hand held at its original initial pose to isolate right-grasp feasibility",
            floating_runtime=True, no_retiming=True)
    oc = int(model.joint("wooden_spoon_free").qposadr[0])
    initial_rotation = Rotation.from_quat(q[0, oc + 3:oc + 7], scalar_first=True)
    pose = q[0].copy()
    pose[columns[:3]] = q[0, oc:oc + 3] + initial_rotation.apply(local_p)
    pose[columns[3:6]] = continuous_euler_xyz(Rotation.from_matrix((initial_rotation * local_r).as_matrix()[None]),
                                            q[0, columns[3:6]])[0]
    pose[columns[6:]] = closed
    np.savez_compressed(directory / "anchor_pose.npz", qpos=pose)
    lift_frames = np.flatnonzero(q[:, oc + 2] - q[0, oc + 2] > .02)
    if not len(lift_frames):
        raise ValueError("No lift in unchanged source target")
    lift = lift_frames[0] / fps
    duration = (len(q) - 1) / fps
    release = (4.8 if episode == 15 else 5.7) if release is None else release
    times = dict(start=max(.05, lift - 1.4), hover=lift - .8 + close_shift,
                 contact=lift - .5 + close_shift, closed=lift - .2 + close_shift,
                 transport=lift + .2 + close_shift, release=min(release, duration - .4))
    spec = dict(episode=name, scene=str(scene), reference=str(seed_reference),
                original_reference=str(inp / "tracking_reference.npz"), source=manifest["source"],
                geometry=str(inp / "metric_geometry.json"), manifest=str(inp / "input_manifest.json"),
                object_name="wooden_spoon", side="right", anchor_pose=str(directory / "anchor_pose.npz"),
                open_fingers=opened.tolist(), times=times, hover_offset=[0., 0., .06],
                horizon_s=min(duration, lift + 2.), score_hold_start_s=times["closed"],
                score_track_start_s=lift + .3, donor=str(DONOR), donor_anchor_s=anchor,
                donor_to_target_registration=registration, unchanged_object_targets=True,
                left_hand_held_at_initial_pose=park_left,
                semantics="Actual measured object-local donor grasp; robot-only approach, closure, transport and release proposal")
    write(directory / "spec.json", spec)
    experiment = PickupExperiment(spec)
    parameters = np.zeros(experiment.dim)
    commands, timing = experiment.commands(parameters)
    object_columns = np.concatenate([np.arange(address, address + 7) for address in experiment.objects.values()])
    if not np.array_equal(commands[:, object_columns], q[:, object_columns]):
        raise AssertionError("Transfer changed object targets")
    reference = TR.save_reference(directory / "reference.npz", model, commands, original["labels"], fps,
                                  original["segment"], original["segments"],
                                  source="Rigid-geometry registered measured grasp proposal; HaWoR prior outside right-hand pickup window",
                                  floating_runtime=True, no_retiming=True)
    pickup_diagnostic = experiment.evaluate(parameters, return_trace=True)
    trace = pickup_diagnostic.pop("trace", None)
    if trace is not None:
        np.savez_compressed(directory / "pickup_trace.npz", trace=trace,
            columns=np.asarray(["time_s", "thumb_gap_m", "finger_gap_m", "both_contact", "qualified_contact",
                                "unsupported", "height_over_start_m", "object_position_error_m", "object_rotation_error_rad",
                                "hand_table_force_n", "wrist_error_m", "other_object_error_m", "self_penetration_m", "hand_penetration_m"]))
    execution = execute_reference(scene, reference, directory / "execution")
    metrics = evaluate_files(manifest["source"], execution["trajectory"], inp / "input_manifest.json",
                             execution["scene"], inp / "metric_geometry.json")
    result = dict(episode=name, input_dir=str(inp), reference=str(reference), execution=execution, metrics=metrics,
                  pickup_diagnostic=pickup_diagnostic, donor=str(DONOR), donor_anchor_s=anchor,
                  timing=timing, registration=registration["best"], learned_policy=False,
                  left_hand_held_at_initial_pose=park_left,
                  all_original_object_targets_unchanged=True, original_initial_state_unchanged=True,
                  submission_ready=False)
    write(directory / "result.json", result)
    print(json.dumps(dict(episode=name, output=str(directory), mp_sr=metrics["reference_proxy_mp_sr"],
                          objects=metrics["official_style_reference_proxy"]["objects"], pickup=pickup_diagnostic)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--anchor", type=float, default=3.5)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--close-shift", type=float, default=0.)
    parser.add_argument("--release", type=float)
    parser.add_argument("--park-left", action="store_true")
    run(**vars(parser.parse_args()))
