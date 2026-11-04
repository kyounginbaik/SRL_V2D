#!/usr/bin/env python3
"""Use HaWoR observed grips to bridge occlusion, with a measured initial cup grip.

Only robot home/commands change. The video begins with the cup already held;
its free-body initial state and every object target remain the supplied values.
"""
from pathlib import Path
import argparse,json,os,sys,xml.etree.ElementTree as ET
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE,LEGACY,ROOT,write
from spoon_transfer import rigid_alignment
sys.path.insert(0,str(LEGACY/'code'));sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from prepare_calibrated_cup_pickups import donor
from tool_library.common import hand as H,tracking as TR
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)

def run(args):
    inp=BASE/'physics_collision_v2/episodes/episode_000020';out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    baseline=json.loads((BASE/'physics_collision_v2/runs/baseline_v1/episode_000020/baseline/result.json').read_text())
    manifest=json.loads((inp/'input_manifest.json').read_text());scene=Path(baseline['execution']['scene'])
    m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(inp/'tracking_reference.npz',m)
    original=ref['qpos'];Q=original.copy();fps=float(ref['fps']);times=np.arange(len(Q))/fps
    measured=donor(LEGACY/'refinement_20261017/cup_observed_retention/episode_000010/a1.9_clear0.015','left','blue_cup',2.5)
    if not measured['qualified_opposing_contact']:raise ValueError('Cupdonor lacks opposingcontact')
    old=json.loads((LEGACY/'episodes/episode_000010/input_manifest.json').read_text())
    T,registration=rigid_alignment(Path(old['source'])/'assets/blue_cup.obj',Path(manifest['source'])/'assets/blue_cup.obj',out/'cup_registration.json')
    p=T[:3,:3]@measured['object_from_measured_wrist_p']+T[:3,3]
    r=Rotation.from_matrix(T[:3,:3])*Rotation.from_quat(measured['object_from_measured_wrist_q'],scalar_first=True)
    cols=m.jnt_qposadr[hand_joints(m,'left')].astype(int);a=int(m.joint('blue_cup_free').qposadr[0])
    objr=Rotation.from_quat(original[:,a+3:a+7],scalar_first=True)
    axis=np.array(json.loads((inp/'metric_geometry.json').read_text())['objects']['blue_cup']['axis_local']);axis/=np.linalg.norm(axis)
    target=objr[0].inv().apply(Q[0,cols[:3]]-Q[0,a:a+3]);u=p-axis*(axis@p);u/=np.linalg.norm(u);v=target-axis*(axis@target);v/=np.linalg.norm(v)
    angle=np.arctan2(axis@np.cross(u,v),u@v);sym=Rotation.from_rotvec(axis*angle)
    p=sym.apply(p);r=sym*r
    w=np.ones(len(times)) if args.safe_cup_release else 1-smooth((times-args.cup_release)/.4)
    targetp=original[:,a:a+3]+objr.apply(np.tile(p,(len(Q),1)))
    Q[:,cols[:3]]+=(targetp-Q[:,cols[:3]])*w[:,None]
    prev=Rotation.from_euler('XYZ',Q[:,cols[3:6]]);corrected=prev*Rotation.from_rotvec((prev.inv()*objr*r).as_rotvec()*w[:,None])
    Q[:,cols[3:6]]=continuous_euler_xyz(corrected,Q[0,cols[3:6]])
    fingers=np.array(measured['measured_finger_q']);fingers+=np.clip(np.array(measured['commanded_finger_q'])-fingers,-.04,.04)
    Q[:,cols[6:]]+=(fingers-Q[:,cols[6:]])*w[:,None]
    if args.safe_cup_release:
        rig=H.HandRig.build(m,'robot_left_','left');d=mujoco.MjData(m)
        d.qpos[:]=Q[0];mujoco.mj_forward(m,d);targets={}
        for finger in H.FINGERS:
            pos=d.site_xpos[rig.fingertip_sites[finger]].copy()
            radial=objr[0].inv().apply(pos-Q[0,a:a+3]);radial-=axis*(axis@radial)
            radial/=max(np.linalg.norm(radial),1e-8)
            targets[finger]=pos+.015*objr[0].apply(radial)
        H.finger_ik(m,d,rig,targets,iters=160);opened=d.qpos[cols[6:]].copy()
        opening=smooth((times-args.cup_release)/.35)
        Q[:,cols[6:]]+=(opened-Q[:,cols[6:]])*opening[:,None]
        local_retreat=p-axis*(axis@p);local_retreat/=np.linalg.norm(local_retreat)
        retreat=objr.apply(np.tile(.08*local_retreat,(len(Q),1)))+np.array([0,0,.03])
        Q[:,cols[:3]]+=retreat*smooth((times-args.cup_release-.35)/.5)[:,None]
    # HaWoR actually observes the right grip near the end of the occlusion.
    # Keep its local wrist/finger configuration through the unobserved interval.
    rc=m.jnt_qposadr[hand_joints(m,'right')].astype(int);pa=int(m.joint('water_pitcher_free').qposadr[0]);k=round(args.right_anchor*fps)
    pr=Rotation.from_quat(original[:,pa+3:pa+7],scalar_first=True)
    lp=pr[k].inv().apply(original[k,rc[:3]]-original[k,pa:pa+3]);lr=pr[k].inv()*Rotation.from_euler('XYZ',original[k,rc[3:6]])
    w=smooth((times-args.grip_start)/.45)*(1-smooth((times-args.pitcher_release)/.4))
    tp=original[:,pa:pa+3]+pr.apply(np.tile(lp,(len(Q),1)))
    Q[:,rc[:3]]+=(tp-Q[:,rc[:3]])*w[:,None]
    prev=Rotation.from_euler('XYZ',Q[:,rc[3:6]]);corrected=prev*Rotation.from_rotvec((prev.inv()*pr*lr).as_rotvec()*w[:,None])
    Q[:,rc[3:6]]=continuous_euler_xyz(corrected,Q[0,rc[3:6]])
    Q[:,rc[6:]]+=(original[k,rc[6:]]-Q[:,rc[6:]])*w[:,None]
    robot=m.jnt_qposadr[m.actuator_trnid[:,0]].astype(int)
    Q[:,robot]=np.clip(Q[:,robot],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    objects=np.setdiff1d(np.arange(m.nq),robot)
    assert np.array_equal(Q[:,objects],original[:,objects])
    tree=ET.parse(scene);home=tree.getroot().find('keyframe/key')
    initial=np.array([float(x) for x in home.get('qpos').split()]);original_xml=initial.copy();initial[robot]=Q[0,robot]
    # Preserve the exact XML object values. MuJoCo normalizes quaternion copies
    # internally, so comparing raw XML against model.key_qpos is inappropriate.
    assert np.array_equal(initial[objects],original_xml[objects])
    home.set('qpos',' '.join(f'{x:.17g}' for x in initial));home.set('ctrl',' '.join(f'{x:.17g}' for x in initial[robot]))
    newscene=out/'controller_scene.xml';tree.write(newscene,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(newscene));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    penetrations=[]
    for c in data.contact:
        if c.dist<-.001:
            penetrations.append(dict(bodies=[model.body(int(model.geom_bodyid[g])).name for g in [c.geom1,c.geom2]],depth_mm=float(-1000*c.dist)))
    write(out/'initial_state_audit.json',dict(robot_initial_state_refined=True,object_initial_states_unchanged=True,penetrations=penetrations))
    path=TR.save_reference(out/'reference.npz',model,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='HaWoR observed pitcher grip across occlusion; measured initial cup grip; robot-only refinement',floating_runtime=True)
    execution=execute_reference(newscene,path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    write(out/'result.json',dict(episode=inp.name,input_dir=str(inp),reference=str(path),execution=execution,metrics=metrics,
        robot_initial_state_refined=True,object_initial_states_unchanged=True,original_object_targets_unchanged=True,
        cup_donor=measured,pitcher_anchor_seconds=args.right_anchor,safe_cup_release=args.safe_cup_release,learned_policy=False,submission_ready=False))
    print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--right-anchor',type=float,default=7.5);p.add_argument('--grip-start',type=float,default=.8)
    p.add_argument('--cup-release',type=float,default=8.7);p.add_argument('--pitcher-release',type=float,default=8.8)
    p.add_argument('--safe-cup-release',action='store_true')
    run(p.parse_args())
