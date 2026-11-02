"""Same tape, two simulators: where do they part, and is the grip already marginal there?"""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from fastsim import FastSim
model,run,world=sys.argv[1],Path(sys.argv[2]),int(sys.argv[3]);z=np.load(run/'physics.npz');r=np.load(run/'references.npz');cap=int(z['native_capture_world_ids'][world]);grp=int(z['world_reference_group'][cap]);C=z['native_qpos'][:,world];s=FastSim(model,run/'physics.npz');n=len(C);F=s.replay(r['joint_q'][grp],r['joint_target'][grp],n);pose=z['object_pose'][:,cap,0];i=pose[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(pose[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(pose[:,10:14]);q=s.qo
print('frame | fast err cm/deg | canon err cm/deg | obj gap mm | Rwrist gap cm | fingers max rad')
for f in list(range(40,n,20)):
    def err(Q):
        return 100*float(np.linalg.norm(Q[f,q:q+3]-gp[f])),float(np.degrees((R.from_quat(Q[f,q+3:q+7][[1,2,3,0]]).inv()*gr[f]).magnitude()))
    a=err(F);b=err(C);print(f,'|',round(a[0],1),round(a[1]),'|',round(b[0],1),round(b[1]),'|',round(1e3*float(np.linalg.norm(F[f,q:q+3]-C[f,q:q+3])),1),'|',round(100*float(np.linalg.norm(F[f,29:32]-C[f,29:32])),2),'|',round(float(np.abs(F[f,36:58]-C[f,36:58]).max()),3))
gap=np.linalg.norm(F[:,q:q+3]-C[:,q:q+3],axis=1);print('first object gap >5mm',int(np.flatnonzero(gap>.005)[0]) if (gap>.005).any() else None,'>20mm',int(np.flatnonzero(gap>.02)[0]) if (gap>.02).any() else None,'>50mm',int(np.flatnonzero(gap>.05)[0]) if (gap>.05).any() else None)
mean=lambda Q:(100*np.linalg.norm(Q[:,q:q+3]-gp[:n],axis=1).mean(),np.degrees((R.from_quat(Q[:,q+3:q+7][:,[1,2,3,0]]).inv()*gr[:n]).magnitude()).mean());print('episode mean fast',np.round(mean(F),2),'canon',np.round(mean(C),2))
