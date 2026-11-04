"""The arm not holding the object follows the human's other hand, and backs off to its ready pose whenever that would touch anything."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,mujoco,pyarrow as pa,pyarrow.parquet as pq,yam_loop as Y,yam_refine as F
from scipy.signal import savgol_filter
CLEAR=.012;ARM,FINGER=3.*.9/Y.FPS,4./Y.FPS;RAMP=int(1.5*Y.FPS)
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,required=True);p.add_argument('--parquet',required=True);a=p.parse_args()
Y.ONSET_FLOOR=1.4;t=Y.YamTask(a.episode,a.side,a.object);t.set_stage('real');m=t.m;d=mujoco.MjData(m);m.geom_margin[:]=.02;o='left' if a.side=='right' else 'right';T=pq.read_table(a.parquet);r=T.to_pylist()[0];q=np.array(r['robot_joint_positions']);mp=F.mapping(m,r['robot_joint_names']);N=len(q)
cols=[i for i,n in enumerate(r['robot_joint_names']) if n.startswith((o+'_','L_arm_j' if o=='left' else 'R_arm_j'))];adr=[mp[i] for i in cols];rdy=t.base_q[adr];lim=np.array([ARM if 'arm_j' in r['robot_joint_names'][i] else FINGER for i in cols])
mine=np.array([m.body(b).name.startswith(o+'_') for b in range(m.nbody)]);robot=np.array([m.body(b).name.startswith(('left_','right_')) for b in range(m.nbody)])
import collections,shutil
why=collections.Counter()
def touches(x,f,log=False):
    d.qpos[:]=t.base_q;d.qpos[adr]=x;g=min(f,len(t.gp_raw)-1);d.qpos[t.qo:t.qo+3]=t.gp_raw[g];d.qpos[t.qo+3:t.qo+7]=t.gr_raw[g].as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    for i in range(d.ncon):
        c=d.contact[i];u,v=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        if mine[u]!=mine[v] and not (robot[u] and robot[v]) and c.dist<CLEAR:
            if log:why[m.body(u if not mine[u] else v).name]+=1
            return True
    return False
alpha=[];shutil.copy(a.parquet,str(a.parquet).replace('reference.parquet','reference_both_raw.parquet'));out=np.zeros((N,len(cols)));x=rdy.copy();backed=np.zeros(N,bool);hit=0
for f in range(N):
    u=min(1.,f/RAMP);want=rdy+u*(q[f,cols]-rdy)
    if touches(want,f,True):
        # Go as far toward the human's pose as stays clear, rather than all the way home.
        lo_,hi_=0.,1.
        for _ in range(6):
            mid=.5*(lo_+hi_)
            if touches(x+mid*(want-x),f):hi_=mid
            else:lo_=mid
        want=x+lo_*(want-x);backed[f]=True;alpha.append(lo_)
    step=x+np.clip(want-x,-lim,lim)
    if touches(step,f):
        back=x+np.clip(rdy-x,-lim,lim);step=back;backed[f]=True
    x=step;out[f]=x
smooth=savgol_filter(out,9,3,axis=0);smooth[0]=rdy
for f in range(N):hit+=touches(smooth[f],f)
for k,i in enumerate(cols):q[:,i]=np.clip(smooth[:,k],*m.jnt_range[m.joint(next(j for j in range(m.njnt) if m.jnt_qposadr[j]==adr[k])).id])
names=[m.joint(next(j for j in range(m.njnt) if m.jnt_qposadr[j]==x_)).name for x_ in adr]
np.savez_compressed(Path(a.episode)/'yam/flash/other_arm.npz',joints=np.array(names),q=q[:,cols])
adr_all=np.array(mp);pb=[m.body(f'{x}_arm_{x}_hand_C_MC').id for x in('left','right')];E=np.array(r['ee_pose_w'],float)
for f in range(N):
    d.qpos[:]=t.base_q;d.qpos[adr_all]=q[f];mujoco.mj_kinematics(m,d);E[f,:,:3]=d.xpos[pb];E[f,:,3:]=d.xquat[pb]
r['robot_joint_positions']=q.tolist();r['ee_pose_w']=E.tolist();pq.write_table(pa.Table.from_pylist([r],schema=T.schema),a.parquet)
pid=m.body(f'{o}_arm_{o}_hand_C_MC').id;palm_path=[]
for f in range(N):
    d.qpos[:]=t.base_q;d.qpos[adr_all]=q[f];mujoco.mj_kinematics(m,d);palm_path.append(d.xpos[pid].copy())
palm_path=np.array(palm_path)
ia=[k for k,i in enumerate(cols) if 'arm_j' in r['robot_joint_names'][i]];print(json.dumps(dict(other_arm=o,range_rad=np.ptp(q[:,cols][:,ia],axis=0).round(2).tolist(),backed_off_frames=int(backed.sum()),frames=N,still_touching_frames=int(hit),limited_by=dict(why.most_common(4)),mean_fraction_when_limited=round(float(np.mean(alpha)),2) if alpha else None,hand_travel_cm=round(100*float(np.linalg.norm(np.diff(palm_path,axis=0),axis=1).sum()),1))),flush=True)
