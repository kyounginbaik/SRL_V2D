"""Frame-by-frame at the real-goal break of a verified rigid grasp."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
model,phys,plan,el=sys.argv[1:5];row=json.loads(Path(el).read_text())[int(sys.argv[5])];s=dict(row['sample'],vmax=float(sys.argv[6]),wmax=float(sys.argv[7]));t=Loop(Path(model),Path(phys),Path(plan),'right',0);t.stage='real';t.set_rung('real');m=t.m
H=70;Q,T=t.compile_tape(s,H);t.s.reset(Q[0]);f6=np.zeros(6);ref=None;bq=t.ball_q;sq=t.slide_q;jm=t.s.jmap;qs=t.s.qs
qi=qs[jm[t.ball]];sl=[qs[jm[j]] for j in t.slides];fq=[qs[jm[j]] for j in t.fingers]
print('f | wristPosErr cm wristRotErr deg fingErrMax | objMove cm objTurn handTurn slipDeg slipMm | upF/mg | contacts(N)')
for f in range(H):
    t.s.step(T[f]);d=t.s.d;po=d.qpos[t.qo:t.qo+3].copy();ro=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]);wr=R.from_matrix(d.xmat[t.body].reshape(3,3).copy());wp=d.xpos[t.body].copy()
    if f==36:ref=(po.copy(),ro,wp.copy(),wr,wr.inv().apply(po-wp))
    if f<34:continue
    up=0.;con={}
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]]);mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3]
        if t.obj in (a,b):
            o=a if b==t.obj else b;n=m.body(o).name.split('_')[-2]+'_'+m.body(o).name.split('_')[-1] if o in t.right else '!'+m.body(o).name[:8];con[n]=round(con.get(n,0)+float(f6[0]),1)
            if o in t.right:up+=float((fw if b==t.obj else -fw)@t.up)
    pe=100*float(np.linalg.norm(d.qpos[sq]-Q[f+1][sl]));re=np.degrees((R.from_quat(d.qpos[bq:bq+4][[1,2,3,0]]).inv()*R.from_quat(Q[f+1][qi:qi+4])).magnitude());fe=float(np.abs(d.qpos[t.finger_q]-Q[f+1][fq]).max())
    if ref is None:print(f,'|',round(pe,2),round(float(re),1),round(fe,2),'| up',round(up/t.mg,2),con);continue
    rel=(ref[3].inv()*wr)*(ref[1].inv()*ro).inv();print(f,'|',round(pe,2),round(float(re),1),round(fe,2),'|',round(100*float(np.linalg.norm(po-ref[0])),1),round(float(np.degrees((ro*ref[1].inv()).magnitude())),0),round(float(np.degrees((wr*ref[3].inv()).magnitude())),0),round(float(np.degrees(rel.magnitude())),0),round(1e3*float(np.linalg.norm(wr.inv().apply(po-wp)-ref[4])),0),'|',round(up/t.mg,2),con)
