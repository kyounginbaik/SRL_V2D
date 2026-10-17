"""Forensic audit of actual spoon-grasp contacts, joint limits and self contact."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1];RUN=B/'refinement_20261018/episode_000003/retention_after45'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H

def main(run=None,sample_stride=1):
 global RUN
 if run is not None:RUN=Path(run).resolve()
 m=mujoco.MjModel.from_xml_path(str(RUN/'execution/scene_physics.xml'));z=np.load(RUN/'execution/trajectory.npz');d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']))
 rig=H.HandRig.build(m,'robot_right_','right');obj=m.body('wooden_spoon').id;fingers=list(H.FINGERS);fc=np.array(list(rig.finger_qpos.values()));fa=np.array(list(rig.finger_act.values()));fj=m.actuator_trnid[fa,0];ranges=m.jnt_range[fj]
 bf={b:next((f for f in fingers if f'_{f}_' in m.body(b).name),'palm') for b in rig.body_ids};f6=np.zeros(6);rows=[];pairs={};objectpairs={};sampled={};maxerr=0.;qmin=np.full(22,np.inf);qmax=-qmin;capfrac=np.zeros(22);limit_samples=0
 recorded_qpos=z['substep_qpos'];recorded_ctrl=z['substep_ctrl']
 for k,ctrl in enumerate(recorded_ctrl):
  d.ctrl[:]=ctrl;mujoco.mj_step(m,d);maxerr=max(maxerr,float(np.abs(d.qpos-recorded_qpos[k]).max()));t=float(d.time)
  if not (2.0<=t<=13.):continue
  qmin=np.minimum(qmin,d.qpos[fc]);qmax=np.maximum(qmax,d.qpos[fc]);capfrac+=np.abs(d.actuator_force[fa])>=.99*np.max(np.abs(m.actuator_forcerange[fa]),axis=1);limit_samples+=1
  if k%sample_stride!=sample_stride-1:continue
  forces={f:0. for f in fingers};padforces={f:0. for f in fingers};patches=[];tableforce=0.;selfdepth=0.
  for i,c in enumerate(d.contact):
   a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);ra,rb=a in bf,b in bf
   if ra and rb:
    key=' | '.join(sorted([m.geom(c.geom1).name,m.geom(c.geom2).name]));old=pairs.get(key,dict(max_penetration_m=0.,contact_substeps=0,max_normal_force_n=0.))
    mujoco.mj_contactForce(m,d,i,f6);old['max_penetration_m']=max(old['max_penetration_m'],-float(c.dist));old['max_normal_force_n']=max(old['max_normal_force_n'],float(f6[0]));old['contact_substeps']+=1;pairs[key]=old;selfdepth=max(selfdepth,-float(c.dist))
   hand=b if a==obj else a if b==obj else -1
   if hand in bf:
    mujoco.mj_contactForce(m,d,i,f6);f=bf[hand];force=max(0.,float(f6[0]));hg=int(c.geom2 if a==obj else c.geom1);name=m.geom(hg).name
    key=name;old=objectpairs.get(key,dict(normal_force_time_integral_ns=0.,contact_substeps=0,max_normal_force_n=0.,max_penetration_m=0.));old['normal_force_time_integral_ns']+=force*m.opt.timestep*sample_stride;old['contact_substeps']+=1;old['max_normal_force_n']=max(old['max_normal_force_n'],force);old['max_penetration_m']=max(old['max_penetration_m'],-float(c.dist));objectpairs[key]=old
    if f in forces:
     forces[f]+=force
     if 'elastomer' in name:padforces[f]+=force
    if force>.1:patches.append((f,np.array(c.pos),np.array(c.frame[:3])*(1 if a==obj else -1),float(c.friction[0]),name))
   if ra!=rb and m.body(b if ra else a).name=='table':mujoco.mj_contactForce(m,d,i,f6);tableforce+=max(0.,float(f6[0]))
  best=None
  for f,p,n,mu,g in patches:
   if f!='thumb':continue
   for f2,p2,n2,mu2,g2 in patches:
    if f2 in ('thumb','palm'):continue
    line=p-p2;dist=np.linalg.norm(line)
    if dist<1e-6:continue
    line/=dist;threshold=1/np.sqrt(1+min(mu,mu2)**2);margin=min(-.5-n@n2,n@line-threshold,n2@(-line)-threshold)
    if best is None or margin>best['margin']:best=dict(finger=f2,margin=float(margin),normal_dot=float(n@n2),thumb_line_cos=float(n@line),finger_line_cos=float(n2@(-line)),geoms=[g,g2])
  rows.append(dict(time_s=t,normal_forces_n=forces,elastomer_normal_forces_n=padforces,opposed_pair=best,table_normal_force_n=tableforce,max_self_penetration_m=selfdepth))
  if any(abs(t-s)<m.opt.timestep*.1 for s in [2.5,3.,5.,8.,10.,12.5]):
   sampled[str(round(t,1))]=dict(actual_deg=dict(zip(rig.finger_qpos,np.degrees(d.qpos[fc]).tolist())),command_deg=dict(zip(rig.finger_qpos,np.degrees(d.ctrl[fa]).tolist())),tip_position_m={f:d.site_xpos[rig.fingertip_sites[f]].tolist() for f in fingers},contacts=rows[-1])
 qout=np.maximum(qmin-ranges[:,0],0)*0
 for i,key in enumerate(rig.finger_qpos):
  qout[i]=max(ranges[i,0]-qmin[i],qmax[i]-ranges[i,1],0.)
 time=len(rows)*m.opt.timestep;force_summary={}
 for f in fingers:
  vals=np.array([r['normal_forces_n'][f] for r in rows]);pads=np.array([r['elastomer_normal_forces_n'][f] for r in rows]);force_summary[f]=dict(contact_fraction=float(np.mean(vals>.1)),mean_normal_force_n=float(vals.mean()),peak_normal_force_n=float(vals.max()),elastomer_fraction_of_force=float(pads.sum()/max(vals.sum(),1e-15)))
 report=dict(scene=str(RUN/'execution/scene_physics.xml'),trajectory=str(RUN/'execution/trajectory.npz'),window_s=[2.,13.],contact_sample_stride=sample_stride,contact_sample_interval_s=float(m.opt.timestep*sample_stride),replay_max_qpos_error=maxerr,finger_contacts=force_summary,
  qualified_opposed_contact_fraction=float(np.mean([r['opposed_pair'] is not None and r['opposed_pair']['margin']>=0 for r in rows])),
  max_hand_table_force_n=max(r['table_normal_force_n'] for r in rows),max_self_penetration_m=max(r['max_self_penetration_m'] for r in rows),
  joint_limits={key:dict(min_actual_deg=float(np.degrees(qmin[i])),max_actual_deg=float(np.degrees(qmax[i])),range_deg=np.degrees(ranges[i]).tolist(),max_violation_deg=float(np.degrees(qout[i])),motor_force_saturation_fraction=float(capfrac[i]/limit_samples)) for i,key in enumerate(rig.finger_qpos)},
  object_contact_geoms=objectpairs,self_contact_pairs=pairs,sampled_postures=sampled,notes=['The explicit self-contact pairs intentionally use source visible meshes; those pairs can activate visual geoms despite zero generic collision bits.','Only robot motion was replayed. Objects remained unactuated, with exact recorded initial integration state.','Opposed contact geometry is a necessary diagnostic, not a general force-closure proof.'])
 out=RUN/'audit';out.mkdir(exist_ok=True);(out/'grip_quality.json').write_text(json.dumps(report,indent=2)+'\n');(out/'grip_quality_timeseries.json').write_text(json.dumps(rows)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['object_contact_geoms','self_contact_pairs','sampled_postures']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--sample-stride',type=int,default=1);main(**vars(p.parse_args()))
