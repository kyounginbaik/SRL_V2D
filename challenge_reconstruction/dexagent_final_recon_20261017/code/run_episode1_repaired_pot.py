"""Search the pot pickup after preserving the repaired-scene lid pickup."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,OUT,EP
from physics_pickup_clearance_gravity import ClearanceExperiment,optimize,dump
from physics_pickup_search_gravity import H,TR
from wrist_reference import continuous_euler_xyz

def prepare():
    root=OUT/'patch_height094';first=root/'pickup_cem_v1';out=root/'pot_input';out.mkdir(exist_ok=True)
    baseline=json.loads((first/'result.json').read_text());spec=dict(baseline['spec']);scene=Path(baseline['execution']['scene'])
    m=mujoco.MjModel.from_xml_path(str(scene));record=np.load(first/'execution/trajectory.npz')
    anchor=record['qpos'][round(3.5*20)].copy();rig=H.HandRig.build(m,'robot_left_','left')
    wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);fc=np.array(list(rig.finger_qpos.values()))
    oc=int(m.joint('white_pot_free').qposadr[0]);targetR=Rotation.from_quat(anchor[oc+3:oc+7],scalar_first=True)
    donor=B/'refinement_20261018/episode_000003/retention_after45'
    dm=mujoco.MjModel.from_xml_path(str(donor/'execution/scene_physics.xml'));dr=H.HandRig.build(dm,'robot_left_','left')
    dw=np.array([dr.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);df=np.array(list(dr.finger_qpos.values()));do=int(dm.joint('white_pot_free').qposadr[0])
    dz=np.load(donor/'execution/trajectory.npz');q=dz['qpos'][100].copy();R=Rotation.from_quat(q[do+3:do+7],scalar_first=True)
    localp=R.inv().apply(q[dw[:3]]-q[do:do+3]);localR=R.inv()*Rotation.from_euler('XYZ',q[dw[3:]])
    # A donor is only a robot grasp proposal. It never changes object state.
    anchor[wc[:3]]=anchor[oc:oc+3]+targetR.apply(localp)
    anchor[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix([(targetR*localR).as_matrix()]),anchor[wc[3:]])[0]
    anchor[fc]=q[df]
    ap=out/'anchor.npz';np.savez_compressed(ap,qpos=anchor)
    donor_ref=TR.load_reference(donor/'reference.npz',dm)
    up=-m.opt.gravity/np.linalg.norm(m.opt.gravity)
    spec.update(scene=str(scene),reference=str(first/'reference.npz'),side='left',object_name='white_pot',anchor_pose=str(ap),
        open_fingers=donor_ref['qpos'][20,df].tolist(),finger_names=list(rig.finger_qpos),hover_offset=(up*.035).tolist(),
        times=dict(start=3.5,hover=4.1,contact=4.55,closed=4.9,transport=5.2,release=11.0),
        score_hold_start_s=5.7,score_track_start_s=5.7,horizon_s=8.5,
        scale=np.r_[[.010]*3,[.10]*3,[.012]*3,[.08]*3,[.12]*22,[.12]*22,[.010]*3,[.08]*3,[.08,.06,.08]].tolist(),
        provenance=dict(donor=str(donor),donor_frame=100,fixed_initial_scene=str(scene),preserved_prefix_s=3.5,
                        immutable_object_targets=True,immutable_right_hand=True,task='left pot pickup after successful initial lid placement'))
    spec.pop('clearance_time_s',None);spec.pop('clearance_min_wrist_z_m',None)
    dump(out/'spec.json',spec);return spec

def run(smoke=False,samples=96,iterations=16,workers=16):
    spec=prepare();root=OUT/'patch_height094'
    if smoke:
        exp=ClearanceExperiment(spec);Q,_=exp.commands(np.zeros(65))
        unchanged=np.setdiff1d(np.arange(exp.m.nq),exp.cols)
        assert np.array_equal(Q[:,unchanged],exp.Q[:,unchanged]);assert np.array_equal(Q[:exp.start_frame+1],exp.Q[:exp.start_frame+1])
        report=exp.evaluate(np.zeros(65),return_trace=True);trace=report.pop('trace');np.savez_compressed(root/'pot_input/smoke_trace.npz',trace=trace)
        dump(root/'pot_input/smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:optimize(spec,root/'pot_cem_v1',samples=samples,iterations=iterations,workers=workers,seed=413)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int,default=96)
    p.add_argument('--iterations',type=int,default=16);p.add_argument('--workers',type=int,default=16);run(**vars(p.parse_args()))
