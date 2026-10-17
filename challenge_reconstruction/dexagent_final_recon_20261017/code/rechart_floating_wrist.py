"""Test a nonsingular fixed frame for the floating wrist's virtual Euler joints.

The hand geometry, reference world poses, object states, time grid, actuator
gains/caps and finger joint bounds are preserved. The virtual rotational axes
and corresponding generalized coordinates change, so this is an explicitly
versioned robot model test, not a claim of identical controller dynamics.
"""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_joint_limits import audit

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def transformed(q,cols,frame):
    old=np.atleast_2d(q[:,cols])
    return continuous_euler_xyz(frame.inv()*Rotation.from_euler('XYZ',old))


def run(episode,chart,source=None,output=None):
    name=f'episode_{episode:06d}'
    source=Path(source).resolve() if source else B/'refinement_20261018/expanded_batch/retention'/name
    result=json.loads((source/'result.json').read_text())
    scene=Path(result['execution']['scene']);old=mujoco.MjModel.from_xml_path(str(scene))
    ref=TR.load_reference(result['reference'],old);original=ref['qpos'];Q=original.copy()
    inp=Path(result['input_dir']) if 'input_dir' in result else Path(result['spec']['manifest']).parent
    manifest=json.loads((inp/'input_manifest.json').read_text())
    cols=np.array([old.joint('robot_right_right_rot_'+a).qposadr[0] for a in 'xyz'])
    aids=np.array([old.actuator('robot_right_right_rot_'+a+'_position').id for a in 'xyz'])
    rotations=Rotation.from_euler('XYZ',original[:,cols])
    rng=np.random.default_rng(3500+episode)
    choices=[Rotation.identity(),rotations.mean(),rotations[0],*Rotation.random(512,random_state=rng)]
    scored=[]
    for frame in choices:
        a=(frame.inv()*rotations).as_euler('XYZ')
        scored.append(float(np.abs(np.cos(a[:,1])).min()))
    # Pick separated good charts, rather than four nearly identical samples.
    selected=[]
    for i in np.argsort(scored)[::-1]:
        frame=choices[i]
        if all((frame.inv()*choices[j]).magnitude()>.6 for j in selected):selected.append(int(i))
        if len(selected)==4:break
    frame=choices[selected[chart]];Q[:,cols]=transformed(original,cols,frame)
    out=(Path(output).resolve() if output else B/'refinement_20261018/wrist_chart'/name)/f'chart_{chart}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    tree=ET.parse(scene);node=tree.find(".//body[@name='robot_right_right_base_roll']")
    if node is None or not np.allclose(old.body('robot_right_right_base_roll').quat,[1,0,0,0]):
        raise ValueError('Expected a zero-offset original rotational base frame')
    node.set('quat',' '.join(format(v,'.17g') for v in frame.as_quat(scalar_first=True)))
    for key in tree.findall('.//keyframe/key'):
        q=np.fromstring(key.get('qpos',''),sep=' ')
        if len(q)!=old.nq:raise ValueError('Expected explicit complete initial qpos')
        q[cols]=transformed(q[None],cols,frame)[0]
        key.set('qpos',' '.join(format(v,'.17g') for v in q))
        velocity=np.fromstring(key.get('qvel',''),sep=' ')
        if len(velocity) and np.any(velocity):raise ValueError('Nonzero initial velocity needs an explicit Jacobian transform')
        ctrl=np.fromstring(key.get('ctrl',''),sep=' ')
        if len(ctrl):
            ctrl[aids]=continuous_euler_xyz(frame.inv()*Rotation.from_euler('XYZ',ctrl[aids][None]))[0]
            key.set('ctrl',' '.join(format(v,'.17g') for v in ctrl))
    changed=out/'scene_recharted.xml';tree.write(changed,encoding='unicode')
    new=mujoco.MjModel.from_xml_path(str(changed))
    if not np.array_equal(old.actuator_gainprm,new.actuator_gainprm) or not np.array_equal(old.actuator_forcerange,new.actuator_forcerange):
        raise ValueError('Actuator gains or force caps changed')
    if not np.array_equal(old.jnt_range,new.jnt_range):raise ValueError('Joint bounds changed')
    od,nd=mujoco.MjData(old),mujoco.MjData(new)
    physical=[i for i in range(old.nbody) if 'robot_right_right_' in old.body(i).name and '_base_' not in old.body(i).name]
    maxp=maxr=0.
    for a,b in zip(original,Q):
        od.qpos[:]=a;nd.qpos[:]=b;mujoco.mj_kinematics(old,od);mujoco.mj_kinematics(new,nd)
        maxp=max(maxp,float(np.max(np.abs(od.xpos[physical]-nd.xpos[physical]))))
        maxr=max(maxr,float(np.max(np.abs(od.xmat[physical]-nd.xmat[physical]))))
    if maxp>1e-10 or maxr>1e-10:raise ValueError('Reference hand poses changed')
    other=np.setdiff1d(np.arange(old.nq),cols)
    if not np.array_equal(original[:,other],Q[:,other]):raise ValueError('Nonwrist references changed')
    mujoco.mj_resetDataKeyframe(old,od,0);mujoco.mj_resetDataKeyframe(new,nd,0)
    mujoco.mj_kinematics(old,od);mujoco.mj_kinematics(new,nd)
    if np.max(np.abs(od.xpos[physical]-nd.xpos[physical]))>1e-10:raise ValueError('Initial physical hand pose changed')
    target=TR.save_reference(out/'reference.npz',new,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Equivalent hand world poses in fixed nonsingular virtual wrist chart',floating_runtime=True)
    validation=dict(chart_quat_wxyz=frame.as_quat(scalar_first=True).tolist(),
        minimum_abs_cos_pitch_before=float(np.abs(np.cos(original[:,cols[1]])).min()),
        minimum_abs_cos_pitch_after=float(np.abs(np.cos(Q[:,cols[1]])).min()),
        peak_euler_rate_before_rad_s=float(np.abs(np.diff(original[:,cols],axis=0)).max()*ref['fps']),
        peak_euler_rate_after_rad_s=float(np.abs(np.diff(Q[:,cols],axis=0)).max()*ref['fps']),
        maximum_fk_position_error_m=maxp,maximum_fk_rotation_matrix_error=maxr,
        same_world_hand_reference_poses=True,nonwrist_reference_unchanged=True,
        object_initial_states_unchanged=True,physical_initial_hand_pose_unchanged=True,
        actuator_gains_and_caps_unchanged=True,finger_joint_limits_unchanged=True,
        virtual_robot_coordinate_axes_changed=True,controller_dynamics_claimed_identical=False)
    dump(out/'kinematic_validation.json',validation)
    execution=execute_reference(changed,target,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'audit/actual_joint_limits.json',audit(execution['scene'],execution['trajectory']))
    dump(out/'result.json',dict(episode=name,input_dir=str(inp),reference=str(target),execution=execution,
        metrics=metrics,kinematic_validation=validation,source=str(source),learned_policy=False))
    print(json.dumps(dict(episode=name,chart=chart,validation=validation,mp_sr=metrics['reference_proxy_mp_sr'],
        objects=metrics['official_style_reference_proxy']['objects'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True)
    p.add_argument('--chart',type=int,choices=range(4),required=True)
    p.add_argument('--source');p.add_argument('--output');run(**vars(p.parse_args()))
