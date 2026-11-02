"""Record the final tape's whole episode in nominal physics for the renderer."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--tape',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--onset-floor',type=float,default=2.);a=p.parse_args();Y.ONSET_FLOOR=a.onset_floor
t=Y.YamTask(a.episode,a.side,a.object);t.set_stage('real');t.nominal_physics();m,d=t.m,t.d;s=json.loads(a.tape.read_text());s=s[0] if isinstance(s,list) else s;s=s['sample'];H=t.frames-1;C,_=t.compile(s,H);F=min(H,len(C));mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d);X=[]
for f in range(F):
    d.ctrl[:]=C[f]
    for _ in range(Y.SUBSTEPS):mujoco.mj_step(m,d)
    X.append(d.qpos.copy())
X=np.array(X);P=np.zeros((F,1,t.poses.shape[1],14))
for k,qi in enumerate(t.object_q):P[:,0,k,:3]=X[:,qi:qi+3];P[:,0,k,3:7]=X[:,qi+3:qi+7][:,[1,2,3,0]];P[:,0,k,7:14]=t.poses[:F,k,7:14]
a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'physics.npz',native_qpos=X[:,None],object_pose=P,native_capture_world_ids=np.array([0]),fps=np.array(Y.FPS));ep=np.linalg.norm(X[:,t.qo:t.qo+3]-t.gp_raw[:F],axis=1);er=(R.from_quat(X[:,t.qo+3:t.qo+7][:,[1,2,3,0]])*t.gr_raw[:F].inv()).magnitude();print(json.dumps(dict(mean_cm=round(100*float(ep.mean()),2),mean_deg=round(float(np.degrees(er.mean())),1),worst_cm=round(100*float(ep.max()),1),worst_deg=round(float(np.degrees(er.max())),1),frames_over=int(((ep>.03)|(er>np.radians(30))).sum()),frames=F)))
