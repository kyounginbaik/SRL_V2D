"""Compare actual compiled actuator damping; XML serialization is not sufficient."""
from pathlib import Path
import json
import mujoco
import numpy as np

B = Path(__file__).resolve().parents[1]
ROOT = B.parents[1]


def audit_model(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    rows = {}
    for aid in range(m.nu):
        jid = int(m.actuator_trnid[aid, 0])
        name = m.joint(jid).name
        if not any('_' + f + '_' in name for f in ['thumb', 'index', 'middle', 'ring', 'pinky']):
            continue
        name = name.removeprefix('robot_left_').removeprefix('robot_right_')
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j].mean()
        mujoco.mj_forward(m, d)
        d.ctrl[:] = d.actuator_length
        mujoco.mj_forward(m, d)
        force0 = float(d.actuator_force[aid])
        d.qvel[m.jnt_dofadr[jid]] = 0.1
        mujoco.mj_forward(m, d)
        rows[name] = dict(actuator_name=m.actuator(aid).name,
            gain=m.actuator_gainprm[aid].tolist(), bias=m.actuator_biasprm[aid].tolist(),
            transmission=int(m.actuator_trntype[aid]), gear=m.actuator_gear[aid].tolist(),
            torque_change_at_positive_0p1_rad_s=float((d.actuator_force[aid] - force0) * m.actuator_gear[aid, 0]),
            generalized_damping_nm_s_rad=float(-(d.actuator_force[aid] - force0) * m.actuator_gear[aid, 0] / 0.1))
    return rows


def main():
    paths = dict(source_left=ROOT/'robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/left_sharpawave.xml',
                 source_right=ROOT/'robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/right_sharpawave.xml',
                 prepared=B/'episodes/episode_000003/stage2/scene.xml',
                 frozen=B/'refinement_20261018/episode_000003/retention_after45/execution/scene_physics.xml')
    models = {name:audit_model(path) for name,path in paths.items()}
    source = models['source_left'] | models['source_right']
    assert len(source) == 44
    comparisons = {}
    for name in ['prepared', 'frozen']:
        assert set(source) == set(models[name])
        comparisons[name] = dict(max_gain_difference=max(float(np.max(np.abs(np.array(source[j]['gain'])-models[name][j]['gain']))) for j in source),
            max_bias_difference=max(float(np.max(np.abs(np.array(source[j]['bias'])-models[name][j]['bias']))) for j in source),
            max_gear_difference=max(float(np.max(np.abs(np.array(source[j]['gear'])-models[name][j]['gear']))) for j in source),
            all_44_finger_actuators_dissipative=all(models[name][j]['generalized_damping_nm_s_rad'] > 0 for j in source))
    report = dict(mujoco_version=mujoco.__version__,model_paths={k:str(v) for k,v in paths.items()},
        comparisons=comparisons,actuators=models,
        interpretation='Source dampratio=0.9 compiles to negative velocity bias. The positive .9 appearing in saved XML is damping-ratio serialization, not runtime anti-damping. All 44 finger servos produce opposing generalized torque for positive joint speed; source, prepared and frozen runtime coefficients are identical.')
    out=B/'refinement_20261018/episode_000003/retention_after45/audit/compiled_finger_damping.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),comparisons=comparisons),indent=2))


if __name__ == '__main__':
    main()
