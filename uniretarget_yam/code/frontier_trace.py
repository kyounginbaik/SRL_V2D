"""Why does this tape stop? Goal, hand-commanded and achieved object, grip and slip per frame."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
model,physics,plan,ms=sys.argv[1:5];t=Loop(Path(model),Path(physics),Path(plan),'right');t.set_rung('real');r=json.loads(Path(ms).read_text())['result'];s=r['sample'];m=t.m
print('sample vmax',round(s['vmax'],3),'wmax',round(s['wmax'],3),'press',round(s['press'],4),'edits',len(s.get('edits',())),'edit frames',[e['frame'] for e in s.get('edits',())])
H=int(sys.argv[5]) if len(sys.argv)>5 else 110;Q,T=t.compile_tape(s,H);t.s.reset(Q[0]);f6=np.zeros(6);ref=None
print('f goalTurn goalRate cmdLag_deg posErr rotErr handMove objMove slipDeg slipMm upF/mg fingers')
for f in range(H):
    t.s.step(T[f]);d=t.s.d;po=d.qpos[t.qo:t.qo+3].copy();ro=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]);wp=d.xpos[t.body].copy();wr=R.from_matrix(d.xmat[t.body].reshape(3,3).copy());up=0.;names=set();wall=0.
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        if t.obj in (a,b):
            other=a if b==t.obj else b;mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3];fo=fw if b==t.obj else -fw
            if other in t.right:up+=float(fo@t.up);names.add(m.body(other).name.replace('right_sharpa_wave_right_',''))
            else:names.add('!'+m.body(other).name[:14])
    if f==45:ref=(po.copy(),ro,wp.copy(),wr)
    if f>=45 and f%2==0:
        goalp=t.gp_raw[f];goalr=t.gr_raw[f];rel=(ref[3].inv()*wr)*(ref[1].inv()*ro).inv();local=wr.inv().apply(po-wp)-ref[3].inv().apply(ref[0]-ref[2])
        rate=float((t.gr_raw[f]*t.gr_raw[f-2].inv()).magnitude()*t.fps/2)
        print(f,round(float(np.degrees((goalr*t.r0.inv()).magnitude())),0),round(rate,1),'|',round(100*float(np.linalg.norm(po-goalp)),1),round(float(np.degrees((ro.inv()*goalr).magnitude())),0),'|',round(100*float(np.linalg.norm(wp-ref[2])),1),round(100*float(np.linalg.norm(po-ref[0])),1),'|',round(float(np.degrees(rel.magnitude())),1),round(1e3*float(np.linalg.norm(local)),1),'|',round(up/t.mg,2),sorted(names))
