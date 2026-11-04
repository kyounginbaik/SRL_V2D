"""Second lid pickup on the unchanged clock, using the first physical grasp."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,OUT,EP
from physics_pickup_clearance_gravity import ClearanceExperiment,optimize,dump
from physics_pickup_search_gravity import H,TR
from wrist_reference import continuous_euler_xyz
from grasp_geometry_seeds_v4 import bounded_ik
from lid_grasp_seed import contact_penetration

def prepare(base=None,actual=False):
    root=OUT/'patch_height094';first=root/'pickup_cem_v1';base=Path(base or first)
    out=root/('regrasp_actual_input' if actual else 'regrasp_input');out.mkdir(exist_ok=True)
    parent=json.loads((base/'result.json').read_text());best=json.loads((first/'result.json').read_text());spec=dict(best['spec'])
    scene=Path(parent['execution']['scene']);m=mujoco.MjModel.from_xml_path(str(scene))
    z=np.load(base/'execution/trajectory.npz');dz=np.load(first/'execution/trajectory.npz');ref=TR.load_reference(first/'reference.npz',m)
    rig=H.HandRig.build(m,'robot_right_','right');wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);fc=np.array(list(rig.finger_qpos.values()));oc=int(m.joint('white_pot_lid_free').qposadr[0])
    q=z['qpos'][round(10.8*20)].copy();donor=dz['qpos'][50];command=donor if actual else ref['qpos'][50]
    donorR=Rotation.from_quat(donor[oc+3:oc+7],scalar_first=True);R=Rotation.from_quat(q[oc+3:oc+7],scalar_first=True)
    gp=donorR.inv().apply(command[wc[:3]]-donor[oc:oc+3]);gr=donorR.inv()*Rotation.from_euler('XYZ',command[wc[3:]])
    q[wc[:3]]=q[oc:oc+3]+R.apply(gp);q[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix([(R*gr).as_matrix()]),q[wc[3:]])[0];q[fc]=command[fc]
    data=mujoco.MjData(m);data.qpos[:]=q;mujoco.mj_forward(m,data);pen=contact_penetration(m,data,rig,m.body('white_pot_lid').id);ik=None
    if pen['table']>.001 or pen['self']>.003 or pen['other']>.003:
        targets={i:data.site_xpos[rig.fingertip_sites[f]].copy() for i,f in enumerate(('thumb','index'))}
        q,ik=bounded_ik(m,data,rig,q,targets,m.body('white_pot_lid').id)
        data.qpos[:]=q;mujoco.mj_forward(m,data)
    after=contact_penetration(m,data,rig,m.body('white_pot_lid').id)
    anchor=out/'anchor.npz';np.savez_compressed(anchor,qpos=q)
    # Opening is inherited as a small joint-space difference from the successful
    # first acquisition, avoiding an unrelated wide-open source posture.
    firstexp=ClearanceExperiment(best['spec']);x=np.array(best['parameters'])
    opened=np.clip(q[fc]+(firstexp.opened+x[12:34])-(firstexp.closed+x[34:56]),firstexp.limits[6:,0],firstexp.limits[6:,1])
    up=-m.opt.gravity/np.linalg.norm(m.opt.gravity)
    spec.update(scene=str(scene),reference=str(base/'reference.npz'),anchor_pose=str(anchor),open_fingers=opened.tolist(),hover_offset=(up*.06).tolist(),
        times=dict(start=10.8,hover=11.35,contact=11.8,closed=12.05,transport=12.4,release=15.9),
        score_hold_start_s=12.3,score_track_start_s=12.5,horizon_s=15.5,
        scale=np.r_[[.008]*3,[.08]*3,[.012]*3,[.08]*3,[.10]*22,[.10]*22,[.008]*3,[.06]*3,[.08,.05,.08]].tolist(),
        provenance=dict(donor=str(first),donor_time_s=2.5,donor_posture='measured' if actual else 'commanded',base=str(base),preserved_prefix_s=10.8,object_targets_unchanged=True,
                        geometry_before=pen,geometry_after=after,ik=ik,scope='Second lid grasp with fixed original timing and fixed physical initial scene'))
    spec.pop('clearance_time_s',None);spec.pop('clearance_min_wrist_z_m',None);dump(out/'spec.json',spec)
    return spec

def run(smoke=False,base=None,actual=False,samples=64,iterations=12,workers=16):
    spec=prepare(base,actual);root=OUT/'patch_height094';inp=root/('regrasp_actual_input' if actual else 'regrasp_input')
    if smoke:
        exp=ClearanceExperiment(spec);Q,_=exp.commands(np.zeros(65));unchanged=np.setdiff1d(np.arange(exp.m.nq),exp.cols)
        assert np.array_equal(Q[:,unchanged],exp.Q[:,unchanged]);assert np.array_equal(Q[:exp.start_frame+1],exp.Q[:exp.start_frame+1])
        report=exp.evaluate(np.zeros(65),return_trace=True);trace=report.pop('trace');np.savez_compressed(inp/'smoke_trace.npz',trace=trace)
        dump(inp/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:optimize(spec,root/('regrasp_actual_cem_v1' if actual else 'regrasp_cem_v1'),samples=samples,iterations=iterations,workers=workers,seed=513)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--base');p.add_argument('--actual',action='store_true');p.add_argument('--samples',type=int,default=64)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=16);run(**vars(p.parse_args()))
