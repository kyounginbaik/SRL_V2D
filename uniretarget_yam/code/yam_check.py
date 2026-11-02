"""Sanity for the reach numbers: IK self-recovery, and multi-start reachability independent of continuity."""
import sys,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from yam_lib import Arm,wrist_world
READY=np.array([0.,1.3,1.3,-1.2,0.,0.]);BASE={'left':np.array([-.31,.26,0.]),'right':np.array([-.31,-.26,0.])};rng=np.random.default_rng(0)
arm=Arm('right');ok=0;xs=[]
for _ in range(60):
    q=rng.uniform(arm.lo,arm.hi);p,r=arm.fk(q);xs.append(p);best=min((arm.ik(rng.uniform(arm.lo,arm.hi),p,r,it=100)[1:] for _ in range(6)),key=lambda v:v[0]);ok+=best[0]<.01 and best[1]<np.radians(5)
xs=np.array(xs);p,r=arm.fk(READY);print('self-recovery',ok,'/60 | ready palm in base frame',np.round(p,3).tolist(),'palm-down angle deg',round(float(np.degrees(np.arccos(np.clip(r.apply([0,0,1])@np.array([0,0,-1.]),-1,1)))),0),'| random reach radius m p50/p90',np.percentile(np.linalg.norm(xs,axis=1),[50,90]).round(2).tolist())
for spec in sys.argv[1:]:
    ep,plan,side,start=spec.split(',');P,Rs,up,poses,fps=wrist_world(ep,plan,side);start=int(start);fx=np.cross([0,1,0],up);fx/=np.linalg.norm(fx);fy=np.cross(up,fx);M=np.c_[fy,-fx,up];c=poses[:,:,:3].reshape(-1,3).mean(0);floor=float((poses[:,:,:3].reshape(-1,3)@up).min())-.03;arm=Arm(side);base=c+M@BASE[side];base=base-(base@up)*up+(floor+.062)*up;BR=R.from_matrix(M);frames=np.arange(start,len(P),max(1,(len(P)-start)//24));pos=[];rot=[];dist=[]
    for f in frames:
        pt=BR.inv().apply(P[f]-base);rt=BR.inv()*Rs[f];dist.append(float(np.linalg.norm(pt)));out=[arm.ik(rng.uniform(arm.lo,arm.hi),pt,rt,it=60)[1:] for _ in range(12)];a,b=min(out,key=lambda v:v[0]+.05*v[1]);pos.append(a);rot.append(b)
    pos=np.array(pos);rot=np.degrees(rot);dp=(P[frames]-base)@M;print(Path(ep).name[-2:],side,'multi-start: reachable',int(((pos<.01)&(rot<5)).sum()),'/',len(frames),'pos-only ok',int((pos<.01).sum()),'| wrist dist from base m med/max',round(float(np.median(dist)),2),round(float(np.max(dist)),2),'| wrist robot-frame xyz med',np.round(np.median(dp,axis=0),2).tolist(),'| rot err deg p50',round(float(np.median(rot)),0),'| base height above table',round(float(base@up-(floor)),3))
