"""Isolated wrist-PD calibration, preserving object/finger/contact physics.

Full original-timing rollouts are independently replay-verified. Gains are
explicit controller alternatives, not trajectory-dependent actuator boosts.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parents[1] / 'third_party/DexAgent'))
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import repair_reference

VARIANTS = {
    'baseline': (1000., 80., 50., 5.),
    'kp5000_r200_d5': (5000., 80., 200., 5.),
    'kp5000_r200_d10': (5000., 80., 200., 10.),
    'kp10000_r200_d5': (10000., 120., 200., 5.),
    'kp10000_r200_d10': (10000., 120., 200., 10.),
}


def controller_scene(source, destination, variant):
    tree = ET.parse(source); root = tree.getroot()
    # Serialized meshes may be relative to the old scene; resolve explicitly.
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    for element in root.iter():
        file = element.get('file')
        if file and not Path(file).is_absolute():
            element.set('file', str((Path(source).parent / (meshdir if element.tag == 'mesh' else '') / file).resolve()))
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
    poskp, poskv, rotkp, rotkv = VARIANTS[variant]
    changed = []
    for act in root.find('actuator'):
        name = act.get('joint', '')
        if name.endswith(('_pos_x', '_pos_y', '_pos_z')):
            kp, kv = poskp, poskv
        elif name.endswith(('_rot_x', '_rot_y', '_rot_z')):
            kp, kv = rotkp, rotkv
        else:
            continue
        act.set('gainprm', str(kp)); act.set('biasprm', f'0 {-kp} {-kv}')
        changed.append(name)
    assert len(changed) == 12
    tree.write(destination)
    a = mujoco.MjModel.from_xml_path(str(source)); b = mujoco.MjModel.from_xml_path(str(destination))
    physical_fields = ['body_mass', 'body_inertia', 'geom_type', 'geom_size', 'geom_friction',
                       'geom_contype', 'geom_conaffinity', 'pair_geom1', 'pair_geom2',
                       'pair_friction', 'jnt_range', 'actuator_forcerange', 'actuator_ctrlrange']
    assert all(np.array_equal(getattr(a, n), getattr(b, n)) for n in physical_fields)
    finger = [i for i in range(a.nu) if a.joint(int(a.actuator_trnid[i,0])).name not in changed]
    assert np.array_equal(a.actuator_gainprm[finger], b.actuator_gainprm[finger])
    assert np.array_equal(a.actuator_biasprm[finger], b.actuator_biasprm[finger])
    return dict(wrist_translation_kp=poskp, wrist_translation_kv=poskv,
                wrist_rotation_kp=rotkp, wrist_rotation_kv=rotkv,
                force_caps_unchanged=True, finger_controller_unchanged=True,
                object_and_collision_physics_unchanged=True)


def tracking_audit(scene, trajectory, reference):
    model = mujoco.MjModel.from_xml_path(str(scene))
    actual = dict(np.load(trajectory)); ref = dict(np.load(reference))
    q, v, command = actual['qpos'], actual['qvel'], ref['qpos']
    fps = float(ref['fps']); results = {}
    for side in ('left', 'right'):
        prefix = f'robot_{side}_{side}_'
        joint_ids = [j for j in range(model.njnt) if model.joint(j).name.startswith(prefix)]
        wrist = [model.joint(prefix + k).id for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
        finger = [j for j in joint_ids if j not in wrist]
        wa = model.jnt_qposadr[wrist]; fa = model.jnt_qposadr[finger]
        epos = q[:,wa[:3]] - command[:,wa[:3]]
        # Cartesian rotation discrepancy, avoiding Euler angle branch artifacts.
        er = (Rotation.from_euler('XYZ', command[:,wa[3:]]).inv() * Rotation.from_euler('XYZ',q[:,wa[3:]])).magnitude()
        pe = np.linalg.norm(epos, axis=1)
        finger_errors = np.abs(q[:,fa] - command[:,fa])
        pvel = np.gradient(command[:,wa[:3]], 1/fps, axis=0)
        largest = np.argsort(np.mean(finger_errors, axis=0))[-5:][::-1]
        out_of_bounds=[]
        for j in joint_ids:
            a=model.jnt_qposadr[j];lo,hi=model.jnt_range[j]
            if model.jnt_limited[j] and np.any((command[:,a]<lo-1e-6)|(command[:,a]>hi+1e-6)):
                out_of_bounds.append(model.joint(j).name)
        results[side]=dict(mean_position_error_m=float(pe.mean()), p95_position_error_m=float(np.quantile(pe,.95)),
            max_position_error_m=float(pe.max()), max_position_error_time_s=float(pe.argmax()/fps),
            mean_position_bias_m=epos.mean(axis=0).tolist(),
            mean_orientation_error_deg=float(np.rad2deg(er).mean()), max_orientation_error_deg=float(np.rad2deg(er).max()),
            max_orientation_error_time_s=float(er.argmax()/fps),
            mean_finger_abs_error_deg=float(np.rad2deg(finger_errors).mean()),
            worst_finger_errors_deg={model.joint(finger[i]).name:float(np.rad2deg(finger_errors[:,i]).mean()) for i in largest},
            max_reference_wrist_speed_m_s=float(np.linalg.norm(pvel,axis=1).max()),
            max_euler_step_deg=float(np.rad2deg(np.abs(np.diff(command[:,wa[3:]],axis=0))).max()),
            out_of_bounds_reference_joints=out_of_bounds)
    return results


def continuous_wrist_reference(model, source, destination):
    """Repair only equivalent Euler branches, preserving every world wrist pose.

    DexAgent's helper omits 2pi wraps on pitch. Crossing pitch=pi can therefore
    command a false simultaneous pi roll/yaw turn even for smooth SO(3) motion.
    """
    return repair_reference(model, source, destination)


def run(episode, variant, output_root, continuous_angles=False):
    episode=f'episode_{int(episode):06d}'
    out=Path(output_root)/episode/variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'calibration_report.json').exists():
        return json.loads((out/'calibration_report.json').read_text())
    source=BASE/'runs/pilot_v2'/episode/'seed_44/baseline/scene_physics.xml'
    scene=out/'controller_scene.xml'
    controller=controller_scene(source,scene,variant)
    reference=BASE/'episodes'/episode/'tracking_reference.npz'
    if continuous_angles:
        repaired=out/'reference_continuous.npz'
        controller['continuous_angle_repair']=continuous_wrist_reference(mujoco.MjModel.from_xml_path(str(scene)),reference,repaired)
        reference=repaired
    begin=time.monotonic()
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(BASE.parents[0]/'final_recon'/episode,execution['trajectory'],
        BASE/'episodes'/episode/'input_manifest.json',execution['scene'])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    audit=tracking_audit(execution['scene'],execution['trajectory'],reference)
    report=dict(episode=episode,variant=variant,controller=controller,
                wall_seconds=time.monotonic()-begin,execution=execution,metrics=metrics,tracking=audit)
    (out/'calibration_report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,required=True)
    parser.add_argument('--variant',choices=list(VARIANTS),required=True)
    parser.add_argument('--output-root',type=Path,default=BASE/'refinement_20261017/controller_sweep')
    parser.add_argument('--continuous-angles',action='store_true')
    args=parser.parse_args();report=run(args.episode,args.variant,args.output_root,args.continuous_angles)
    print(json.dumps(report,indent=2))
