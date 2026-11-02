"""Compare tapes on one nominal rollout: tracking per moment, contact depth and finger effort while held."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from yam_loop import YamTask,FPS,SUBSTEPS
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--tape',nargs='+',required=True);p.add_argument('--onset-floor',type=float,default=2.);a=p.parse_args();import yam_loop;yam_loop.ONSET_FLOOR=a.onset_floor
t=YamTask(a.episode,a.side,0);t.set_stage('real');t.nominal_physics();m,d=t.m,t.d;H=t.frames-1
for path in a.tape:
    s=json.loads(Path(path).read_text());s=s[0] if isinstance(s,list) else s;s=s['sample'];C,_=t.compile(s,H);mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d);E=[];A=[];depth=[];normal=[];eff=[];f6=np.zeros(6)
    for f in range(min(H,len(C))):
        d.ctrl[:]=C[f]
        for _ in range(SUBSTEPS):mujoco.mj_step(m,d)
        E.append(100*float(np.linalg.norm(d.qpos[t.qo:t.qo+3]-t.gp_raw[f])));A.append(float(np.degrees((t.gr_raw[f]*R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]).inv()).magnitude())))
        if f>t.onset+10:
            dd=[0.];ff=0.
            for i in range(d.ncon):
                c=d.contact[i];x,y=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                if t.obj in(x,y) and (x in t.right or y in t.right):dd.append(-float(c.dist));mujoco.mj_contactForce(m,d,i,f6);ff+=float(f6[0])
            depth.append(max(dd));normal.append(ff);eff.append(float(np.max(np.abs(d.actuator_force[t.act_f])/m.actuator_forcerange[t.act_f,1])))
    E,A=np.array(E),np.array(A);print(json.dumps(dict(tape=str(path)[-40:],mean_cm=round(E.mean(),2),mean_deg=round(A.mean(),1),worst_cm=round(E.max(),1),worst_cm_s=round(E.argmax()/FPS,2),worst_deg=round(A.max(),1),worst_deg_s=round(A.argmax()/FPS,2),frames_over=int(((E>3)|(A>30)).sum()),deg_after_4s=round(A[80:].mean(),1),contact_depth_mm_p50_max=[round(1e3*float(np.median(depth)),2),round(1e3*float(max(depth)),2)],grip_normal_N_p50=round(float(np.median(normal)),2),finger_effort_frac_p50_max=[round(float(np.median(eff)),3),round(float(max(eff)),3)])),flush=True)
