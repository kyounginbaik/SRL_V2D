"""Explicit finger-PD alternatives with unchanged mechanical limits and references."""
from pathlib import Path
import argparse, json, sys, time, xml.etree.ElementTree as ET
import numpy as np
import mujoco

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent'))
from floating_runtime import execute_reference
from calibrate_controller import tracking_audit
from evaluate_mp_sr import evaluate_files
from tool_library.common import hand as H


def finger_gain_scene(source, destination, multiplier, side=None):
    """Scale stiffness and servo damping only; preserve caps, friction and inertia."""
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    for element in root.iter():
        value = element.get('file')
        if value and not Path(value).is_absolute():
            element.set('file', str((Path(source).parent / (meshdir if element.tag == 'mesh' else '') / value).resolve()))
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
    old = mujoco.MjModel.from_xml_path(str(source))
    suffixes = tuple('_' + s for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z'))
    changed = []
    for actuator in root.find('actuator'):
        joint = actuator.get('joint', '')
        if not joint.startswith('robot_') or joint.endswith(suffixes):
            continue
        if side is not None and not joint.startswith(f'robot_{side}_{side}_'):
            continue
        aid = old.actuator(actuator.get('name')).id
        gain = old.actuator_gainprm[aid].copy()
        bias = old.actuator_biasprm[aid].copy()
        gain[0] *= multiplier
        bias[1] *= multiplier
        bias[2] *= np.sqrt(multiplier)
        actuator.set('gainprm', ' '.join(map(str, gain)))
        actuator.set('biasprm', ' '.join(map(str, bias)))
        changed.append(aid)
    assert len(changed) == (44 if side is None else 22), len(changed)
    tree.write(destination)
    new = mujoco.MjModel.from_xml_path(str(destination))
    unchanged_fields = ['body_mass','body_inertia','body_pos','body_quat','body_gravcomp',
        'geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype',
        'geom_conaffinity','mesh_vert','mesh_face','pair_geom1','pair_geom2','pair_friction',
        'jnt_range','dof_frictionloss','dof_damping','actuator_forcerange','actuator_ctrlrange',
        'actuator_trnid','key_qpos','qpos0']
    for field in unchanged_fields:
        assert np.array_equal(getattr(old,field),getattr(new,field)), field
    wrist = np.setdiff1d(np.arange(old.nu), changed)
    assert np.array_equal(old.actuator_gainprm[wrist],new.actuator_gainprm[wrist])
    assert np.array_equal(old.actuator_biasprm[wrist],new.actuator_biasprm[wrist])
    return dict(finger_stiffness_multiplier=multiplier, side=side or 'both',
        finger_servo_damping_multiplier=float(np.sqrt(multiplier)),
        unchanged_model_fields=unchanged_fields, wrist_controller_unchanged=True,
        changed_actuators=[old.actuator(a).name for a in changed])


def grasp_audit(scene, trajectory):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    record = dict(np.load(trajectory))
    rig = H.HandRig.build(model,'robot_right_','right')
    lid = model.body('white_pot_lid').id
    fps = 1. / np.median(np.diff(record['timestamps_seconds']))
    rows = []
    for t in np.arange(1.4,2.41,.1):
        frame = round(t*fps)
        data.qpos[:] = record['qpos'][frame]
        data.qvel[:] = record['qvel'][frame]
        data.ctrl[:] = record['ctrl'][frame]
        mujoco.mj_forward(model,data)
        contacts = []
        for contact in data.contact:
            a,b = int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
            if a in rig.body_ids or b in rig.body_ids:
                contacts.append(dict(bodies=[model.body(a).name,model.body(b).name],
                    penetration_m=max(0.,-float(contact.dist)),
                    lid_contact=lid in (a,b)))
        fingers = {}
        for aid in range(model.nu):
            jid = int(model.actuator_trnid[aid,0]); name=model.joint(jid).name
            if name.startswith('robot_right_right_') and ('thumb' in name or 'index' in name):
                adr=model.jnt_qposadr[jid]
                fingers[name]=dict(error_deg=float(np.rad2deg(record['reference'][frame,adr]-data.qpos[adr])),
                    force_nm=float(data.actuator_force[aid]),
                    force_cap_nm=float(np.max(np.abs(model.actuator_forcerange[aid]))))
        rows.append(dict(time_s=float(t),contacts=contacts,fingers=fingers))
    return rows


def run(task):
    candidate=task//2; factor=(2.,4.)[task%2]
    source=B/f'refinement_20261017/lid_pressure/episode_000001/candidate_{candidate:02d}_press_1'
    out=B/f'refinement_20261017/finger_gain/episode_000001/candidate_{candidate:02d}_kp{factor:g}'
    out.mkdir(parents=True,exist_ok=True)
    scene=out/'controller_scene.xml'
    begin=time.monotonic()
    controller=finger_gain_scene(source/'execution/scene_physics.xml',scene,factor)
    reference=source/'reference.npz'
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,source_reference=str(reference),controller=controller,
        execution=execution,metrics=metrics,
        tracking=tracking_audit(execution['scene'],execution['trajectory'],reference),
        grasp=grasp_audit(execution['scene'],execution['trajectory']),wall_seconds=time.monotonic()-begin)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(candidate=candidate,factor=factor,metrics=metrics),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
