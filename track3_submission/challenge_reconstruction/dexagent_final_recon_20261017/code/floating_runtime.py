"""Explicit floating-Sharpa execution port for the DexAgent experiment.

This is an adapted robot runtime, NOT DexAgent's rigid-arm certification.
Object motion is always MuJoCo contact physics. Only the 56 named hand joints
receive position-servo commands; every integration state/control is recorded.
The unchanged upstream physics_recording_check independently replays those
controls. Its schema constant is scoped to the declared floating embodiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import mujoco
import numpy as np

SCHEMA = "dexagent.floating_sharpa.v1"
WRIST = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
_SCHEMA_LOCK = threading.RLock()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_contract(model):
    """Fail closed on unbounded actuators, object actuation or attachments."""
    issues = []
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "dexagent_robot_schema")
    if marker < 0:
        issues.append("missing explicit floating embodiment schema")
    else:
        start, size = int(model.text_adr[marker]), int(model.text_size[marker])
        value = bytes(model.text_data[start:start + size]).rstrip(b"\x00").decode()
        if value != SCHEMA:
            issues.append(f"expected {SCHEMA}, got {value}")
    expected = set()
    sides = {}
    for side in ("left", "right"):
        head = f"robot_{side}_{side}_"
        joints = [j for j in range(model.njnt) if (model.joint(j).name or "").startswith(head)]
        if len(joints) != 28:
            issues.append(f"{side}: expected 6 wrist and 22 finger joints, got {len(joints)}")
        for k, suffix in enumerate(WRIST):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, head + suffix)
            want = mujoco.mjtJoint.mjJNT_SLIDE if k < 3 else mujoco.mjtJoint.mjJNT_HINGE
            if j < 0 or model.jnt_type[j] != want:
                issues.append(f"{side}: invalid wrist joint {suffix}")
        expected.update(joints)
        sides[side] = {"joint_count": len(joints)}
    actuated = []
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            issues.append(f"actuator {a}: only named hand-joint transmission allowed")
            continue
        j = int(model.actuator_trnid[a, 0]); actuated.append(j)
        if j not in expected:
            issues.append(f"actuator {a} acts outside robot hands: {model.joint(j).name}")
        if not model.actuator_forcelimited[a] or not np.isfinite(model.actuator_forcerange[a]).all():
            issues.append(f"actuator {a} lacks finite force bounds")
        if not model.actuator_ctrllimited[a] or not np.isfinite(model.actuator_ctrlrange[a]).all():
            issues.append(f"actuator {a} lacks finite target bounds")
        if model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE:
            issues.append(f"actuator {a}: dynamic actuators are outside this port's contract")
        if (model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_gainprm[a, 0] <= 0
                or not np.isclose(model.actuator_biasprm[a, 1], -model.actuator_gainprm[a, 0])
                or not np.allclose(model.actuator_gear[a], [1., 0., 0., 0., 0., 0.])):
            issues.append(f"actuator {a}: expected unit-gear joint position servo")
    if set(actuated) != expected or len(actuated) != len(expected):
        issues.append("each hand joint must have exactly one actuator; object actuators forbidden")
    if any("_arm_" in (model.joint(j).name or "") for j in range(model.njnt)):
        issues.append("arm joints are outside this explicitly floating embodiment")
    if np.any(model.body_gravcomp):
        issues.append("gravity compensation must be zero; actuator effort must support hands")
    if model.nmocap:
        issues.append("mocap bodies are forbidden in this physics experiment")
    if model.neq:
        issues.append("this rigid-object experiment permits no equality attachments")
    if not (0 < float(model.opt.timestep) <= .005):
        issues.append("physics timestep must be positive and at most 5 ms")
    return {"ok": not issues, "issues": issues, "schema": SCHEMA,
            "sides": sides, "actuators": model.nu, "objects_actuated": False,
            "timestep_s": float(model.opt.timestep)}


def _assert_contract(model):
    report = model_contract(model)
    if not report["ok"]:
        raise ValueError("floating runtime contract: " + "; ".join(report["issues"]))
    return report


def servo_rollout(model, Qref, fps, qpos_start=None, qvel_start=None,
                  state_start=None, record_substeps=True, **options):
    """Run position servos without state injection or changing scene physics.

    Frame zero is the initial physical state. Remaining reference rows execute
    at their original timestamps (no time stretching). Complete integration
    state can be supplied for exact MPC trial/resume. Qref object columns never
    enter ctrl. record_substeps=False only reduces diagnostic storage.
    """
    from scipy.spatial.transform import Rotation
    from tool_library.common import episode as E
    _assert_contract(model)
    for name in ("eq_inactive", "rules", "latched", "mechanism_state_start"):
        if options.get(name):
            raise ValueError(f"{name}: mechanisms are not supported by this rigid-object adapter")
    if any(options.get(k) is not None for k in ("finger_force_nm", "wrist_force_n", "wrist_torque_nm", "finger_kp")):
        raise ValueError("runtime gains/force limits are frozen in scene.xml; per-trial changes are forbidden")
    if options.get("settle_s", 0) or options.get("ctrl_lead_s", 0) or options.get("balance_grip", False):
        raise ValueError("settling, time lead and adaptive gain changes are excluded")
    Qref = np.asarray(Qref, dtype=float)
    if Qref.ndim != 2 or Qref.shape[1] != model.nq or not len(Qref) or not np.isfinite(Qref).all():
        raise ValueError("reference must be finite [T, model.nq]")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0 or fps * model.opt.timestep > 1:
        raise ValueError("fps must be positive and no faster than physics")
    data, target, measured = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(model, state_spec)
    if state_start is not None:
        raw = np.asarray(state_start, float)
        if raw.shape != (state_size,) or not np.isfinite(raw).all():
            raise ValueError("invalid complete initial integration state")
        mujoco.mj_setState(model, data, raw, state_spec)
        if qpos_start is not None and not np.array_equal(data.qpos, qpos_start):
            raise ValueError("qpos_start contradicts integration state")
        if qvel_start is not None and not np.array_equal(data.qvel, qvel_start):
            raise ValueError("qvel_start contradicts integration state")
    else:
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        if qpos_start is not None:
            data.qpos[:] = np.asarray(qpos_start, float)
        if qvel_start is not None:
            data.qvel[:] = np.asarray(qvel_start, float)
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("external force injection is forbidden")
    mujoco.mj_forward(model, data)
    initial_qpos, initial_qvel, initial_act = data.qpos.copy(), data.qvel.copy(), data.act.copy()
    initial_state = np.zeros(state_size)
    mujoco.mj_getState(model, data, initial_state, state_spec)
    initial_time = float(data.time)
    parts = E.robot_parts(model)
    ids = np.arange(model.nu)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]].astype(int)
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange.T
    T, dt = len(Qref), float(model.opt.timestep)
    Q = np.empty((T, model.nq)); V = np.empty((T, model.nv)); C = np.empty((T, model.nu))
    times = np.empty(T); tracking = np.zeros((T, 2)); forces = np.zeros((T, 2))
    pen = np.zeros(T); objpen = np.zeros(T); wrench = np.zeros((T, 2))
    contacts = []; sq = []; sv = []; sc = []; st = []
    force6 = np.zeros(6); tangent = np.zeros(model.nv)
    robot = {b for b in range(model.nbody) if (model.body(b).name or "").startswith("robot_")}
    side_of_body = {b: si for si, s in enumerate(("left", "right")) for b in parts[s]["rig"].body_ids}
    slide_aids = [a for a in ids if model.jnt_type[int(model.actuator_trnid[a, 0])] == mujoco.mjtJoint.mjJNT_SLIDE]
    rot_aids = [a for a in ids if any((model.joint(int(model.actuator_trnid[a, 0])).name or "").endswith(s) for s in WRIST[3:])]

    def diagnose(t):
        touched, net = set(), np.zeros((2, 3))
        for ci in range(data.ncon):
            contact = data.contact[ci]
            b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            r1, r2 = b1 in robot, b2 in robot
            depth = max(0., -float(contact.dist))
            if r1 != r2:
                pen[t] = max(pen[t], depth)
                hand, obj = (b1, b2) if r1 else (b2, b1)
                si = side_of_body.get(hand)
                if si is not None:
                    touched.add((("left", "right")[si], model.body(obj).name))
                    mujoco.mj_contactForce(model, data, ci, force6)
                    net[si] += contact.frame.reshape(3, 3).T @ force6[:3] * (1 if r1 else -1)
            elif not r1 and not r2:
                objpen[t] = max(objpen[t], depth)
        forces[t] = np.maximum(forces[t], np.linalg.norm(net, axis=1))
        wrench[t] = np.maximum(wrench[t], [np.max(np.abs(data.actuator_force[slide_aids]), initial=0.),
                                           np.max(np.abs(data.actuator_force[rot_aids]), initial=0.)])
        return sorted(touched)

    unstable = None
    completed = 0
    for t in range(T):
        previous, current = round(max(0, t - 1) / (fps * dt)), round(t / (fps * dt))
        n = current - previous
        for k in range(n):
            # Interpolate on generalized coordinates; only robot joints command actuators.
            mujoco.mj_differentiatePos(model, tangent, 1., Qref[t - 1], Qref[t])
            target.qpos[:] = Qref[t - 1]
            mujoco.mj_integratePos(model, target.qpos, tangent, (k + 1) / n)
            data.ctrl[:] = np.clip(target.qpos[qadr], ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)
            if record_substeps:
                sq.append(data.qpos.copy()); sv.append(data.qvel.copy()); sc.append(data.ctrl.copy())
                st.append(float(data.time) - initial_time)
            # Do not call mj_forward here: the exact integrator warmstart state is preserved.
            if record_substeps:
                diagnose(t)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
                unstable = {"frame": t, "reason": "nonfinite state or BADQACC reset"}
                break
        Q[t], V[t], C[t], times[t] = data.qpos, data.qvel, data.ctrl, float(data.time) - initial_time
        # Kinematics only on copied data: diagnostics cannot perturb simulation warmstart.
        measured.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, measured)
        target.qpos[:] = Qref[t]; mujoco.mj_kinematics(model, target)
        for side, part in parts.items():
            body = part["ee_body"]
            p = np.linalg.norm(target.xpos[body] - measured.xpos[body])
            R = target.xmat[body].reshape(3, 3).T @ measured.xmat[body].reshape(3, 3)
            tracking[t] = np.maximum(tracking[t], [p, np.degrees(Rotation.from_matrix(R).magnitude())])
        contacts.append(diagnose(t)); completed = t + 1
        if unstable:
            break
    final_state = np.zeros(state_size); mujoco.mj_getState(model, data, final_state, state_spec)
    sl = slice(0, completed)
    return {"qpos": Q[sl], "qvel": V[sl], "ctrl": C[sl], "time_s": times[sl],
            "initial_qpos": initial_qpos, "initial_qvel": initial_qvel, "initial_act": initial_act,
            "initial_state": initial_state, "final_state": final_state, "state_spec": int(state_spec),
            "initial_time_s": initial_time, "final_qpos": data.qpos.copy(), "final_qvel": data.qvel.copy(),
            "track_pos_m": tracking[sl, 0], "track_rot_deg": tracking[sl, 1],
            "hand_pen_m": pen[sl], "object_pen_m": objpen[sl], "hand_object_force_n": forces[sl],
            "wrist_force_n": wrench[sl, 0], "wrist_torque_nm": wrench[sl, 1],
            "contacts": contacts, "unstable": unstable, "frames_simulated": completed,
            "robot_schema": SCHEMA, "eq_inactive": [], "released": [], "latched": {}, "latch_events": [],
            "mechanism_initial": {}, "mechanism_final": {}, "gains": {"source": "frozen_scene"},
            "substep_qpos": np.asarray(sq).reshape(-1, model.nq),
            "substep_qvel": np.asarray(sv).reshape(-1, model.nv),
            "substep_ctrl": np.asarray(sc).reshape(-1, model.nu), "substep_time_s": np.asarray(st)}


def verify_recording(scene, trajectory):
    """Run upstream independent replay, adapting only the schema identifier."""
    from tool_library.common import rigid_robot as RR
    from tool_library.verifiers.physics_recording_check import physics_recording_check
    with _SCHEMA_LOCK:
        old = RR.SCHEMA
        try:
            RR.SCHEMA = SCHEMA
            return physics_recording_check(str(scene), str(trajectory))
        finally:
            RR.SCHEMA = old


def floating_runtime_check(scene, trajectory):
    model = mujoco.MjModel.from_xml_path(str(scene))
    report = model_contract(model)
    with np.load(trajectory, allow_pickle=False) as archive:
        if str(archive.get("robot_schema", "")) != SCHEMA:
            report["issues"].append("recording does not identify this floating embodiment")
        if str(archive.get("scene_sha256", "")) != _sha(scene):
            report["issues"].append("recording belongs to a different scene")
        keys = {"timestamps_seconds", "object_names", "object_positions", "object_quaternions_wxyz"}
        if keys.intersection(archive.files):
            if not keys.issubset(archive.files):
                report["issues"].append("incomplete canonical object pose export")
            else:
                data = mujoco.MjData(model)
                names = list(archive["object_names"].astype(str))
                valid = archive["object_positions"].shape == (len(archive["qpos"]), len(names), 3)
                valid &= archive["object_quaternions_wxyz"].shape == (len(archive["qpos"]), len(names), 4)
                valid &= np.allclose(archive["timestamps_seconds"], archive["time_s"], atol=1e-12, rtol=0)
                if not valid:
                    report["issues"].append("canonical object arrays/timing do not match recording")
                else:
                    for t, q in enumerate(archive["qpos"]):
                        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
                        for i, name in enumerate(names):
                            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                            if body < 0 or not np.allclose(data.xpos[body], archive["object_positions"][t, i], atol=1e-10, rtol=0):
                                report["issues"].append(f"canonical position disagrees with physics at {t}, {name}")
                                break
                            aq, bq = data.xquat[body], archive["object_quaternions_wxyz"][t, i]
                            if min(np.linalg.norm(aq - bq), np.linalg.norm(aq + bq)) > 1e-10:
                                report["issues"].append(f"canonical orientation disagrees with physics at {t}, {name}")
                                break
                        if report["issues"]:
                            break
    report["ok"] = not report["issues"]
    return report


def _freeze(scene, out):
    from tool_library.common import episode as E
    from tool_library.common import handgeom
    source = mujoco.MjModel.from_xml_path(str(scene)); _assert_contract(source)
    frozen = out / "scene_physics.xml"
    # Our imported robot already carries its exact URDF collision meshes.
    # Upstream's palm/thumb hull augmentation repairs THEIR simpler primitives;
    # adding it here duplicates contacts. Preserve imported geometry exactly.
    with _SCHEMA_LOCK:
        original = handgeom.add_hand_hulls
        try:
            handgeom.add_hand_hulls = lambda spec: []
            E.build_execution_model(str(scene), out_xml=str(frozen))
        finally:
            handgeom.add_hand_hulls = original
    # Serialize and reload BEFORE execution so replay uses exactly the same bytes.
    model = mujoco.MjModel.from_xml_path(str(frozen)); _assert_contract(model)
    for side in ("left", "right"):
        prefix = f"robot_{side}_"
        before = [g for g in range(source.ngeom)
                  if (source.body(int(source.geom_bodyid[g])).name or "").startswith(prefix)]
        after = [g for g in range(model.ngeom)
                 if (model.body(int(model.geom_bodyid[g])).name or "").startswith(prefix)]
        old_names = {source.geom(g).name for g in before if source.geom(g).name}
        new_names = {model.geom(g).name for g in after}
        # The upstream self-contact builder names formerly anonymous visual
        # geoms; names may be added, while physical geometry may not be added.
        if len(before) != len(after) or not old_names.issubset(new_names):
            raise ValueError("execution compiler changed exact source hand collision/visual geometry")
    return model, frozen


def _write(model, frozen, out, diag, Qref, labels, fps, extra=None):
    from tool_library.common import episode as E
    archive = {k: v for k, v in diag.items() if isinstance(v, np.ndarray)}
    archive.update(robot_schema=np.array(SCHEMA), scene_sha256=np.array(_sha(frozen)),
                   fps=float(fps), state_spec=int(diag["state_spec"]),
                   reference=Qref[:len(diag["qpos"])], labels=np.asarray(labels[:len(diag["qpos"])]),
                   joint_names=np.asarray(E.joint_names(model)),
                   actuator_names=np.asarray([model.actuator(a).name for a in range(model.nu)]))
    objects = E.object_free_joints(model)
    object_names = list(objects)
    positions = np.empty((len(diag["qpos"]), len(objects), 3))
    quaternions = np.empty((len(diag["qpos"]), len(objects), 4))
    data = mujoco.MjData(model)
    for t, q in enumerate(diag["qpos"]):
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        for i, name in enumerate(object_names):
            body = model.body(name).id
            positions[t, i] = data.xpos[body]
            quaternions[t, i] = data.xquat[body] / np.linalg.norm(data.xquat[body])
    archive.update(timestamps_seconds=diag["time_s"].copy(),
                   object_names=np.asarray(object_names), object_positions=positions,
                   object_quaternions_wxyz=quaternions)
    path = out / "trajectory.npz"; np.savez_compressed(path, **archive)
    replay = verify_recording(frozen, path)
    contract = floating_runtime_check(frozen, path)
    report = {"execution_completed": diag["unstable"] is None and len(diag["qpos"]) == len(Qref),
              "physics_verified": replay["ok"] and contract["ok"], "task_success": None,
              "scope": "DexAgent planning with explicit floating-Sharpa physics port; MP-SR evaluated separately",
              "scene": str(frozen), "trajectory": str(path), "frames": len(diag["qpos"]),
              "duration_s": float(diag["time_s"][-1]), "fps": fps,
              "max_hand_pen_m": float(diag["hand_pen_m"].max()),
              "max_object_pen_m": float(diag["object_pen_m"].max()),
              "max_wrist_position_tracking_m": float(diag["track_pos_m"].max()),
              "max_wrist_rotation_tracking_deg": float(diag["track_rot_deg"].max()),
              "peak_wrist_force_n": float(diag["wrist_force_n"].max()),
              "peak_wrist_torque_nm": float(diag["wrist_torque_nm"].max()),
              "unstable": diag["unstable"], "contacts_end": diag["contacts"][-1],
              "runtime_contract": contract, "independent_control_replay": replay,
              "hashes": {"scene": _sha(frozen), "trajectory": _sha(path), "adapter": _sha(__file__)}}
    report.update(extra or {})
    (out / "execution_report.json").write_text(json.dumps(report, indent=2, default=_json))
    return report


def execute_reference(scene, reference, out_dir, fps=None, qpos_start=None, expected_duration_s=None):
    """Execute a fixed-time proposal; first object state comes from scene home."""
    from tool_library.common import episode as E
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / "trajectory.npz").exists():
        raise FileExistsError("keep previous trials; select a fresh out_dir")
    model, frozen = _freeze(scene, out)
    if isinstance(reference, (str, Path)):
        with np.load(reference, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
    else:
        ref = reference
    Qref = np.asarray(ref["qpos"], float)
    if "joint_names" in ref:
        names, wanted = list(np.asarray(ref["joint_names"]).astype(str)), E.joint_names(model)
        if set(names) != set(wanted) or len(names) != len(set(names)):
            raise ValueError("reference joint identities do not match the frozen scene")
        Qref = Qref[:, [names.index(n) for n in wanted]]
    fps = float(fps if fps is not None else ref.get("fps", 20.))
    duration = (len(Qref) - 1) / fps
    if expected_duration_s is not None and abs(duration - float(expected_duration_s)) > .5 / fps:
        raise ValueError("proposal duration differs from original reference timing")
    start = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, start, 0)
    if qpos_start is not None and not np.allclose(qpos_start, start.qpos, rtol=0, atol=1e-10):
        raise ValueError("trial initial state is fixed to scene keyframe, not a candidate's object poses")
    labels = list(ref.get("labels", ["both: reference"] * len(Qref)))
    if len(labels) != len(Qref):
        raise ValueError("one label per reference frame required")
    diag = servo_rollout(model, Qref, fps, qpos_start=start.qpos.copy())
    return _write(model, frozen, out, diag, Qref, labels, fps,
                  {"reference_source": str(reference) if isinstance(reference, (str, Path)) else "in_memory",
                   "requested_duration_s": duration, "no_retiming": True})


def execute_plan(scene, plan, out_dir, fps=20., expected_duration_s=None):
    """Use DexAgent keyframe planner, then execute each subgoal from actual state.

    The standard planner and physical outcome checks stay unchanged. No arm IK,
    arm speed retiming, fixed flange claim, or renderer-only animation is used.
    Each subgoal recording is independently replayable; task metric is separate.
    """
    from tool_library.common import outcome as O
    from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
    from tool_library.skills.chain_subgoals import chain_subgoals
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    model, frozen = _freeze(scene, out)
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    subgoals = plan.get("subgoals", plan.get("plan", [])) if isinstance(plan, dict) else plan
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    q, state, complete_state, command = d.qpos.copy(), {}, None, d.qpos.copy()
    records, duration = [], 0.
    for i, subgoal in enumerate(subgoals):
        directory = out / f"subgoal_{i+1:02d}"; directory.mkdir(exist_ok=True)
        planned = plan_subgoal_keyframes(model, subgoal, qpos0=q, state=state, command_qpos=command,
                                         fps=fps, auto_via=False, out_json=str(directory / "keyframes.json"),
                                         max_rate_deg_s=None, max_rate_m_s=None)
        chained = chain_subgoals(model, planned["keyframes"], fps=fps,
                                  max_rate_deg_s=None, max_rate_m_s=None, q0=q)
        diag = servo_rollout(model, chained["qpos"], fps, qpos_start=q, state_start=complete_state)
        labels = chained["labels"]; O.resolve_label_frames(planned["expect"], labels)
        outcome = O.evaluate(model, diag["qpos"], fps, list(range(len(labels))), planned["expect"], diagnostics=diag)
        rec = _write(model, frozen, directory, diag, chained["qpos"], labels, fps,
                     {"subgoal": subgoal, "outcome": outcome, "no_retiming": True})
        records.append(rec); duration += rec["duration_s"]
        q, complete_state, state = diag["final_qpos"], diag["final_state"], planned["state"]
        command = chained["qpos"][-1].copy()
        if not outcome["ok"] or not rec["physics_verified"] or not rec["execution_completed"]:
            break
    report = {"physics_verified": bool(records) and all(r["physics_verified"] for r in records),
              "all_subgoals_passed": len(records) == len(subgoals) and all(r["outcome"]["ok"] for r in records),
              "task_success": None, "duration_s": duration, "subgoals": records,
              "scope": "adapted floating-Sharpa DexAgent subgoal execution; requires separate fixed-time MP-SR"}
    if expected_duration_s is not None:
        report["original_timing_preserved"] = abs(duration - expected_duration_s) <= .5 / fps
    (out / "plan_report.json").write_text(json.dumps(report, indent=2, default=_json))
    return report


def bootstrap():
    """Opt-in override for upstream callers; rigid robot execution is untouched."""
    from tool_library.common import episode as E
    if getattr(E.servo_rollout, "_floating_port", False):
        return
    original = E.servo_rollout
    def dispatch(model, Qref, fps, *args, **kwargs):
        contract = model_contract(model)
        if contract["ok"]:
            return servo_rollout(model, Qref, fps, *args, **kwargs)
        return original(model, Qref, fps, *args, **kwargs)
    dispatch._floating_port = True
    E.servo_rollout = dispatch
