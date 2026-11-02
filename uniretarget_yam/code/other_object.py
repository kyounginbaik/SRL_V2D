"""What moves the object a round isn't refining? Per-frame error, travel, and who touches it."""
import sys,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
e=Path(sys.argv[1]);run=Path(sys.argv[2]);world=int(sys.argv[3]);k=int(sys.argv[4]);m=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));z=np.load(run/'physics.npz');cap=int(z['native_capture_world_ids'][world]);P=z['object_pose'][:,cap];off=z['native_contact_offsets'];W=z['native_contact_worldid'];G=z['native_contact_geom'];F=z['native_contact_newton_force'];Q0=z['native_qpos'][0,0];free=[j for j in range(m.njnt) if m.jnt_type[j]==0];body=lambda o:int(m.jnt_bodyid[min(free,key=lambda j:float(np.linalg.norm(Q0[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-P[0,o,:3])))])
for o in range(P.shape[1]):
    x=P[:,o];i=x[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(x[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(x[:,10:14]);ep=100*np.linalg.norm(x[:,:3]-gp,axis=1);er=np.degrees((R.from_quat(x[:,3:7]).inv()*gr).magnitude());print('obj',o,m.body(body(o)).name[:22],'mass',round(float(m.body_mass[body(o)]),3),'episode mean',round(ep.mean(),2),'cm',round(er.mean(),1),'deg | goal travel',round(100*float(np.linalg.norm(gp-gp[0],axis=1).max()),1),'cm | err by frame',[(f,round(ep[f],1),int(er[f])) for f in range(30,len(ep),20)])
ob=body(k);print('contacts on obj',k,'(force N) per frame:')
name=lambda b:(m.body(b).name.split('_')[0][0].upper()+'_'+m.body(b).name.split('_')[-2]+m.body(b).name.split('_')[-1]) if m.body(b).name.startswith(('left_','right_')) else m.body(b).name[:12]
for f in range(int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7])):
    a,b=off[f],off[f+1];row={}
    for i in np.arange(a,b)[W[a:b]==cap]:
        u,v=int(m.geom_bodyid[G[i,0]]),int(m.geom_bodyid[G[i,1]])
        if ob in (u,v):
            n=name(v if u==ob else u);row[n]=round(row.get(n,0)+float(np.linalg.norm(F[i,:3])),1)
    x=P[f,k];print(' f',f,'pos',np.round(x[:3],3).tolist(),'tilt from start deg',int(np.degrees((R.from_quat(x[3:7])*R.from_quat(P[0,k,3:7]).inv()).magnitude())),dict(sorted(row.items(),key=lambda kv:-kv[1])[:5]))
