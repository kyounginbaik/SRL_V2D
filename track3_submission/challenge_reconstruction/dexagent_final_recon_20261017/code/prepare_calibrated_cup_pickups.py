"""Measured cup-grasp seeds for real calibrated-frame pickup optimization."""
from pathlib import Path
import json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

B=Path(__file__).resolve().parents[1];R17=B/'refinement_20261017';R18=B/'refinement_20261018'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from wrist_reference import continuous_euler_xyz


def donor(path,side,obj,seconds):
    m=mujoco.MjModel.from_xml_path(str(path/'execution/scene_physics.xml'));z=np.load(path/'execution/trajectory.npz');d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']))
    for ctrl in z['substep_ctrl'][:round(seconds/m.opt.timestep)]:d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    rig=H.HandRig.build(m,f'robot_{side}_',side);ob=m.body(obj).id;forces={};normals={};patches=[];support=0.;force=np.zeros(6)
    for i,c in enumerate(d.contact):
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);other=b2 if b1==ob else b1 if b2==ob else -1
        if other<0:continue
        mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]));name=m.body(other).name
        if not name.startswith('robot_'):support+=fn
        if other not in rig.body_ids:continue
        finger=next((f for f in H.FINGERS if f'_{f}_' in name),None)
        if finger is None:continue
        forces[finger]=forces.get(finger,0.)+fn
        if fn>.1:
            normal=np.array(c.frame[:3])*(1 if b1==ob else -1);normals.setdefault(finger,[]).append(normal);patches.append((finger,np.array(c.pos),normal,float(c.friction[0])))
    qualified=False
    for fa,pa,na,ma in patches:
        if fa!='thumb':continue
        for fb,pb,nb,mb in patches:
            if fb=='thumb':continue
            line=pa-pb;length=np.linalg.norm(line)
            if length<1e-5:continue
            line/=length;threshold=1/np.sqrt(1+min(ma,mb)**2)
            qualified|=bool(na@nb<-.5 and na@line>=threshold and nb@(-line)>=threshold)
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));fa=np.array(list(rig.finger_act.values()));oa=int(m.joint(obj+'_free').qposadr[0]);rot=Rotation.from_quat(d.qpos[oa+3:oa+7],scalar_first=True)
    result=dict(source=str(path),time_s=seconds,side=side,object=obj,normal_forces_n=forces,nonrobot_support_force_n=support,qualified_opposing_contact=qualified,
        object_from_measured_wrist_p=rot.inv().apply(d.qpos[wc[:3]]-d.qpos[oa:oa+3]).tolist(),object_from_measured_wrist_q=(rot.inv()*Rotation.from_euler('XYZ',d.qpos[wc[3:]])).as_quat(scalar_first=True).tolist(),
        measured_finger_q=d.qpos[fc].tolist(),commanded_finger_q=d.ctrl[fa].tolist(),finger_names=list(rig.finger_qpos),full_measured_qpos=d.qpos.tolist())
    return result


def main():
    inp=R18/'episode_000010/calibrated_inputs/episodes/episode_000010';out=R18/'episode_000010/calibrated_pickup_inputs';out.mkdir(exist_ok=True)
    scene=R18/'episode_000010/calibrated_baselines/controller_scene.xml';m=mujoco.MjModel.from_xml_path(str(scene));original=TR.load_reference(inp/'tracking_reference.npz',m);Q=original['qpos']
    observed=np.load(R18/'episode_000010/calibrated_baselines/native/execution/trajectory.npz');rest=observed['qpos'][10].copy()
    definitions=[('left','blue_cup',R17/'cup_observed_retention/episode_000010/a1.9_clear0.015',2.5),('right','beige_cup',R17/'feedback_polish/episode_000008/both_0.5',3.5)]
    for side,obj,source,t in definitions:
        info=donor(source,side,obj,t);rig=H.HandRig.build(m,f'robot_{side}_',side);wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));aids=np.array(list(rig.finger_act.values()));limits=m.jnt_range[m.actuator_trnid[aids,0]]
        oa=int(m.joint(obj+'_free').qposadr[0]);rp=rest[oa:oa+3];rr=Rotation.from_quat(rest[oa+3:oa+7],scalar_first=True);lp=np.array(info['object_from_measured_wrist_p']);lr=Rotation.from_quat(info['object_from_measured_wrist_q'],scalar_first=True)
        # Cup axial symmetry permits another robot contact location. Align the
        # measured grasp to the approach side nearest the initial robot wrist.
        home=rr.inv().apply(Q[0,wc[:3]]-rp);angle=np.arctan2(home[1],home[0])-np.arctan2(lp[1],lp[0]);sym=Rotation.from_rotvec([0.,0.,angle]);lp=sym.apply(lp);lr=sym*lr
        anchor=rest.copy();anchor[wc[:3]]=rp+rr.apply(lp);anchor[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix([(rr*lr).as_matrix()]),Q[0,wc[3:]])[0]
        measured=np.array(info['measured_finger_q']);pressure=np.clip(np.array(info['commanded_finger_q'])-measured,-.04,.04);anchor[fc]=np.clip(measured+pressure,limits[:,0]+1e-8,limits[:,1]-1e-8)
        d=mujoco.MjData(m);d.qpos[:]=anchor;mujoco.mj_forward(m,d);active=[f for f,v in info['normal_forces_n'].items() if v>.1]
        if 'thumb' not in active:raise ValueError('Donor lacks thumb contact')
        targets={}
        for finger in active:
            p=d.site_xpos[rig.fingertip_sites[finger]].copy();v=rr.inv().apply(p-rp);v[2]=0;v/=max(np.linalg.norm(v),1e-9);targets[finger]=p+.004*rr.apply(v)
        initial=anchor[fc].copy();low=np.maximum(limits[:,0]+1e-8,initial-.20);high=np.minimum(limits[:,1]-1e-8,initial+.20)
        def residual(x):
            d.qpos[fc]=x;mujoco.mj_forward(m,d)
            return np.r_[np.concatenate([(d.site_xpos[rig.fingertip_sites[f]]-p)/.004 for f,p in targets.items()]),.15*(x-initial)/.1]
        fitted=least_squares(residual,initial,bounds=(low,high),max_nfev=45);opened=fitted.x
        anchor_path=out/f'{side}_anchor.npz';np.savez_compressed(anchor_path,qpos=anchor)
        base=Q.copy()
        if side=='left':
            # Isolate left acquisition first; the final combined test restores
            # and optimizes the other real hand. Objects remain dynamic.
            other=[int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.joint(j).name.startswith('robot_right_')];base[:,other]=Q[0,other]
        reference=TR.save_reference(out/f'{side}_reference.npz',m,base,original['labels'],original['fps'],original['segment'],original['segments'],source='Calibrated cup pickup geometry experiment; immutable original object targets.',floating_runtime=True)
        offset=rr.apply(np.r_[lp[:2]/max(np.linalg.norm(lp[:2]),1e-9)*.04,0.])+np.array([0.,0.,.04])
        spec=dict(scene=str(scene),reference=str(reference),original_reference=str(inp/'tracking_reference.npz'),source=str(B.parents[1]/'challenge_reconstruction/final_recon/episode_000010'),geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),object_name=obj,side=side,anchor_pose=str(anchor_path),open_fingers=opened.tolist(),finger_names=list(rig.finger_qpos),times=dict(start=.25,hover=.75,contact=1.10,closed=1.45,transport=2.20,release=5.5 if side=='left' else 5.),horizon_s=3.2,hover_offset=offset.tolist(),donor=info,robot_grasp_axial_rotation_rad=float(angle),score_hold_start_s=2.0,score_track_start_s=2.3)
        spec['scale']=np.r_[[.008]*3,[.08]*3,[.02]*3,[.12]*3,[.08]*22,[.08]*22,[.012]*3,[.08]*3,[.08,.07,.10]].tolist()
        (out/f'{side}.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps(dict(side=side,donor_contact=info['normal_forces_n'],opposed=info['qualified_opposing_contact'],support_force=info['nonrobot_support_force_n'],grasp_rotation=angle)),flush=True)


if __name__=='__main__':main()
