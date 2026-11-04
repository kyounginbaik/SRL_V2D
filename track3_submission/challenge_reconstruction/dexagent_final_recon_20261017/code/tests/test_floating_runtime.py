"""Runtime evidence tests on the actual episode-1 embodiment, no task claim."""
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import floating_runtime as F


def main(scene):
    from tool_library.common import episode as E
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert F.model_contract(model)["ok"]
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    reference = np.tile(data.qpos, (5, 1))
    wrist = model.jnt_qposadr[model.joint("robot_left_left_pos_x").id]
    reports = []
    for offset in (0., .003, -.003):
        Q = reference.copy(); Q[:, wrist] += np.linspace(0., offset, len(Q))
        result = F.servo_rollout(model, Q, 20)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            report = F._write(model, Path(scene), p, result, Q, ["both: test"] * len(Q), 20.)
            assert report["physics_verified"], report
            z = np.load(p / "trajectory.npz", allow_pickle=False)
            changed = {k: z[k] for k in z.files}
            changed["qpos"] = changed["qpos"].copy(); changed["qpos"][2, wrist] += .001
            np.savez(p / "tampered.npz", **changed)
            assert not F.verify_recording(scene, p / "tampered.npz")["ok"]
            reports.append({"wrist_delta_m": offset, "exact_control_replay": True, "tamper_rejected": True})
    # Object targets have no control channel: malicious object columns cannot move objects.
    alternate = reference.copy()
    for adr in E.object_free_joints(model).values():
        alternate[:, adr:adr+3] += [1., 2., 3.]
    a = F.servo_rollout(model, reference, 20)
    b = F.servo_rollout(model, alternate, 20)
    assert np.array_equal(a["qpos"], b["qpos"])
    assert np.array_equal(a["ctrl"], b["ctrl"])
    # MPC trials keep a full integration state; their own controls can be replayed.
    first = F.servo_rollout(model, reference[:3], 20)
    last = F.servo_rollout(model, reference[2:], 20, state_start=first["final_state"], record_substeps=False)
    assert np.allclose(last["qpos"][0], first["qpos"][-1], atol=0, rtol=0)
    assert np.allclose(last["qpos"][-1], a["qpos"][-1], atol=1e-9, rtol=0)
    assert last["substep_qpos"].shape[0] == 0
    # Changing bounds to infinite/unbounded invalidates the embodiment contract.
    saved = model.actuator_forcelimited.copy()
    model.actuator_forcelimited[0] = False
    assert not F.model_contract(model)["ok"]
    model.actuator_forcelimited[:] = saved
    print(json.dumps({"ok": True, "cases": reports, "object_targets_ignored": True,
                      "full_state_continuation": True, "unbounded_actuator_rejected": True}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
