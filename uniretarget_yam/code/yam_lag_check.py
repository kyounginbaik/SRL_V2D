"""Split spoon error during the carry into reference lag, arm tracking lag and slip in the hand; explain edit size at the palm."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from yam_loop import YamTask,FPS,SUBSTEPS,ARM_SPEED
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--tape',required=True);p.add_argument('--onset-floor',type=float,default=2.);p.add_argument('--from-s',type=float,default=1.5);p.add_argument('--to-s',type=float,default=5.);a=p.parse_args();import yam_loop;yam_loop.ONSET_FLOOR=a.onset_floor
t=YamTask(a.episode,a.side,0);t.set_stage('real');t.nominal_physics();m,d=t.m,t.d;H=t.frames-1;s=json.loads(Path(a.tape).read_text());s=s[0] if isinstance(s,list) else s;s=s['sample'];C,_=t.compile(s,H);wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);hp,hr=t.follow(s,H+1);o_=dict(s);o_['edits']=[];hp0,hr0=t.follow(o_,H+1);ds=mujoco.MjData(m);V=np.gradient(C[:,t.act_arm],1/FPS,axis=0)
mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d);rows=[];rel0=None;v=[];u=[]
sat=lambda:float(np.max(np.abs(d.actuator_force[t.act_arm])/m.actuator_forcerange[t.act_arm,1]))
print(json.dumps(dict(onset=t.onset,lever_palm_to_object_origin_cm=round(100*float(np.linalg.norm(wp-t.o0)),1),vmax=round(s['vmax'],3),wmax=round(s['wmax'],3))))
for f in range(min(H,len(C))):
    d.ctrl[:]=C[f]
    for _ in range(SUBSTEPS):mujoco.mj_step(m,d)
    ds.qpos[:]=d.qpos;ds.qpos[t.q_arm]=C[f,t.act_arm];mujoco.mj_kinematics(m,ds);pt=ds.xpos[t.palm].copy()
    pa,ra=t.pose(d);po=d.qpos[t.qo:t.qo+3].copy();ro=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]);obj_in=ra.inv().apply(po-pa);ang_in=ra.inv()*ro
    if f==t.onset:rel0=(obj_in.copy(),ang_in)
    relh=hr[f]*t.r0.inv();ref=hp[f]+relh.apply(wp-t.o0)
    slip=(100*float(np.linalg.norm(obj_in-rel0[0])),float(np.degrees((ang_in*rel0[1].inv()).magnitude()))) if rel0 else (0.,0.)
    qdot=np.abs(d.qvel[t.v_arm]);v.append(qdot.max());u.append(sat())
    if a.from_s*FPS<=f<=a.to_s*FPS and f%5==0:rows.append(dict(t=round(f/FPS,2),spoon_cm=round(100*float(np.linalg.norm(po-t.gp_raw[f])),1),spoon_deg=round(float(np.degrees((t.gr_raw[f]*ro.inv()).magnitude()))),goal_up=round(100*float((t.gp_raw[f]-t.o0)@t.up),1),spoon_up=round(100*float((po-t.o0)@t.up),1),refpath_vs_goal=round(100*float(np.linalg.norm(hp[f]-t.gp_raw[f])),1),palmcmd_vs_ref=round(100*float(np.linalg.norm(pt-ref)),1),palm_vs_cmd=round(100*float(np.linalg.norm(pa-pt)),1),slip_cm=round(slip[0],1),slip_deg=round(slip[1]),qd=round(float(qdot.max()),2),qd_cmd=round(float(np.abs(V[f]).max()),2),effort=round(u[-1],2)))
for r in rows:print(json.dumps(r))
dr=np.zeros(3);dp=np.zeros(3)
for e in s.get('edits',()):g=np.exp(-.5*((60-e['frame'])/e['width'])**2);dr+=g*np.asarray(e['dr']);dp+=g*np.asarray(e['dp'])
relh=hr[60]*t.r0.inv();rel0h=hr0[60]*t.r0.inv();pal1=hp[60]+relh.apply(wp-t.o0);pal0=hp0[60]+rel0h.apply(wp-t.o0);print(json.dumps(dict(at_3s_edit_dp_cm=round(100*float(np.linalg.norm(dp)),1),edit_dr_deg=round(float(np.degrees(np.linalg.norm(dr))),1),commanded_palm_shift_cm=round(100*float(np.linalg.norm(pal1-pal0)),1),lever_times_angle_cm=round(100*float(np.linalg.norm(wp-t.o0))*float(np.linalg.norm(dr)),1))))
print(json.dumps(dict(peak_arm_speed=round(max(v),2),limit=ARM_SPEED,peak_effort=round(max(u),2))))
