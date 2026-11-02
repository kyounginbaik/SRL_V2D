"""Refined vs original reference, kinematics only: robot at its targets, every object on its first-pose-aligned goal path."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from yam_loop import YamTask
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--tape',type=Path,required=True);p.add_argument('--original',type=Path);p.add_argument('--executed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--onset-floor',type=float,default=2.);a=p.parse_args();import yam_loop;yam_loop.ONSET_FLOOR=a.onset_floor
t=YamTask(a.episode,a.side,a.object);t.set_stage('real');t.nominal_physics();m=t.m;d=mujoco.MjData(m);H=t.frames-1;s=json.loads(a.tape.read_text());s=s[0] if isinstance(s,list) else s;s=s['sample'];o=json.loads(a.original.read_text())['sample'] if a.original else s;o=dict(o);o['edits']=[]
Cr,_=t.compile(s,H);Co,_=t.compile(o,H);z=np.load(a.executed/'physics.npz');Qx=z['native_qpos'][:,0];n=min(len(Cr),len(Co),len(Qx))
def kin(C):
    Q=Qx[:n].copy();adr=m.jnt_qposadr[m.actuator_trnid[:,0]];Q[:,adr]=C[:n];return Q
Qr,Qo=kin(Cr),kin(Co);dp=[];dr=[];b=t.palm
for f in range(n):
    d.qpos[:]=Qr[f];mujoco.mj_kinematics(m,d);p1,r1=d.xpos[b].copy(),R.from_matrix(d.xmat[b].reshape(3,3).copy());d.qpos[:]=Qo[f];mujoco.mj_kinematics(m,d);p2,r2=d.xpos[b].copy(),R.from_matrix(d.xmat[b].reshape(3,3).copy());dp.append(np.linalg.norm(p1-p2));dr.append((r1*r2.inv()).magnitude())
dp,dr=np.array(dp),np.array(dr);a.output.mkdir(parents=True,exist_ok=True);P=z['object_pose'][:n].copy();free=[j for j in range(m.njnt) if m.jnt_type[j]==0]
for k in range(P.shape[2]):
    o=P[:,0,k];al=R.from_quat(o[0,3:7])*R.from_quat(o[0,10:14]).inv();gp=al.apply(o[:,7:10])+o[0,:3]-al.apply(o[0,7:10]);gq=(al*R.from_quat(o[:,10:14])).as_quat();adr=m.jnt_qposadr[min(free,key=lambda j:float(np.linalg.norm(Qx[0,m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-o[0,:3])))];Qr[:,adr:adr+3]=gp;Qr[:,adr+3:adr+7]=gq[:,[3,0,1,2]];P[:,0,k,:3]=gp;P[:,0,k,3:7]=gq
np.savez_compressed(a.output/'physics.npz',native_qpos=Qr[:,None],object_pose=P,native_capture_world_ids=np.array([0]),fps=z['fps']);np.savez_compressed(a.output/'ghost.npz',qpos=Qo,palm_cm=100*dp,palm_deg=np.degrees(dr))
print(json.dumps(dict(frames=n,edits=[dict(frame=e['frame'],width=round(e['width'],1),dp_cm=round(100*float(np.linalg.norm(e['dp'])),1),dr_deg=round(float(np.degrees(np.linalg.norm(e['dr']))),1)) for e in s.get('edits',[])],peak_palm_cm=round(100*float(dp.max()),1),peak_palm_deg=round(float(np.degrees(dr.max())),1),frames_changed=int(((dp>.002)|(dr>np.radians(2))).sum()),by_15=[[f,round(100*float(dp[f]),1),round(float(np.degrees(dr[f])),0)] for f in range(0,n,15)])))
