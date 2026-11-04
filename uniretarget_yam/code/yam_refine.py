"""UniRetarget grasp refinement seeded from an IK reference: contacts start where the reference's fingertips reach the object."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y
from flash_chord.retargeting.surface_contact_search import demand_coverage
SEED={};SIG=(.012,.04)
def mapping(m,names):
    o=[]
    for n in names:
        j=(("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]) if n.startswith(("R_arm_j","L_arm_j")) else (("right" if n.startswith("right") else "left")+"_arm_"+n)
        o.append(int(m.jnt_qposadr[m.joint(j).id]))
    return o
def reference_tips(t,parquet,before=8):
    """Reference fingertips and palm in the object's rest frame, averaged over the frames just before the lift."""
    r=pq.read_table(parquet).to_pylist()[0];q=np.array(r['robot_joint_positions']);mp=mapping(t.m,r['robot_joint_names']);d=t.p;out={x:[] for x in Y.DIGITS};pl=[];pr=[]
    for f in range(max(0,t.onset-before),t.onset+1):
        d.qpos[:]=t.base_q;d.qpos[mp]=q[f];mujoco.mj_kinematics(t.m,d);to=t.gr_raw[f].inv()
        for x in Y.DIGITS:out[x].append(t.gp_raw[0]+t.gr_raw[0].apply(to.apply(t.tip_world(d,x)-t.gp_raw[f])))
        a,b=t.pose(d);pl.append(t.gp_raw[0]+t.gr_raw[0].apply(to.apply(a-t.gp_raw[f])));pr.append((t.gr_raw[0]*to*b).as_quat())
    q_=np.array(pr);q_*=np.sign(q_@q_[0])[:,None]
    shift=t.o0-t.gp_raw[0];palm=R.from_quat(q_.mean(0)/np.linalg.norm(q_.mean(0)))
    if Y.JOIN and Y.JOIN.get('start') is not None:
        # Mid-motion: everything above was expressed against the goal's first pose; carry it to the pose this hand grasps at.
        rot=t.r0*t.gr_raw[0].inv();to_=lambda x:t.o0+rot.apply(x-t.gp_raw[0]);return {x:to_(np.mean(v,0)) for x,v in out.items()},to_(np.mean(pl,0)),rot*palm
    return {x:np.mean(v,0)+shift for x,v in out.items()},np.mean(pl,0)+shift,palm
REACH={};REACH_S,REACH_MIN=45.,.8;OPPOSED=Y.YamTask.propose
def reach_seeds(t,seconds=REACH_S):
    """Palm poses at the object that the arm can carry along the whole goal: grasp where the task stays reachable, not only where the human's hand was."""
    import os,time
    rng=np.random.default_rng(os.getpid());d=t.p;d.qpos[:]=t.base_q;half=np.array(t.q_open)+.35*(t.m.jnt_range[t.fingers,1]-np.array(t.q_open));d.qpos[t.q_f]=half;mujoco.mj_kinematics(t.m,d);a,b=t.pose(d);local={x:b.inv().apply(t.tip_world(d,x)-a) for x in Y.DIGITS};aim=.5*(local['thumb']+local['index'])
    P,N=t.surface(rng,1500);gp,gr=t.task_goal[:2] if hasattr(t,'task_goal') else (t.gp,t.gr);step=Y.ARM_SPEED/Y.FPS;out=[];end=time.monotonic()+seconds
    while time.monotonic()<end and len(out)<24:
        pair=OPPOSED(t,rng,tries=5) if rng.random()<.5 else None
        if pair is not None:
            # Fit the hand to contacts that oppose each other: on a large object a random palm rarely has anything to pinch.
            names_=('thumb','index','middle');G=np.array([pair[x] for x in names_]);L=np.array([local[x] for x in names_]);r,_=R.align_vectors(G-G.mean(0),L-L.mean(0));r=R.from_rotvec(rng.normal(0,.1,3))*r;p=G.mean(0)-r.apply(L.mean(0))
        else:
            i=int(rng.integers(len(P)));r0,_=R.align_vectors([-N[i]],[t.local_normal]);r=R.from_rotvec(N[i]*rng.uniform(-np.pi,np.pi))*r0;r=R.from_rotvec(rng.normal(0,.2,3))*r;p=P[i]-r.apply(aim)
        if not (Y.JOIN and Y.JOIN.get('start') is not None) and r.apply(t.local_normal)@t.natural_palm<Y.NATURAL_PALM_MIN:continue
        q,e1,e2=t.arm_ik(Y.READY+rng.normal(0,.5,6),p,r,steps=60)
        if e1>=.03 or e2>=Y.TOL_R:continue
        hit=c=0
        for f in range(t.onset,t.frames):
            rel=gr[f]*t.r0.inv();q,e1,e2=t.arm_ik(q,gp[f]+rel.apply(p-t.o0),rel*r,steps=10,limit=step);hit+=e1<.03 and e2<Y.TOL_R;c+=1
        if hit/max(1,c)>=REACH_MIN:
            tips_={x:p+r.apply(v) for x,v in local.items()}
            if pair is not None:tips_.update({x:np.asarray(pair[x]) for x in ('thumb','index','middle')})
            out.append((p,r,tips_))
    return out
def propose(self,rng,tries=30):
    """Contacts sampled around the reference's fingertips, not over the whole object."""
    if 'tips' not in SEED:SEED['tips'],SEED['palm_p'],SEED['palm_r']=reference_tips(self,SEED['parquet'])
    if 'seeds' not in REACH:REACH['seeds']=reach_seeds(self)
    self.palm_seed=(SEED['palm_p'],SEED['palm_r']);T=SEED['tips'];self.palm_seed_p=.3;self.seed_kind='reference'
    if REACH['seeds'] and rng.random()<.5:
        p_,r_,T=REACH['seeds'][int(rng.integers(len(REACH['seeds'])))];self.palm_seed=(p_,r_);self.palm_seed_p=.8;self.seed_kind='reach'
    P,N=self.surface(rng,1500);sig=SIG if rng.random()<.7 else (.03,.09)
    def pick(x,mask=None):
        s=float(rng.uniform(*sig));w=np.exp(-(np.linalg.norm(P-T[x],axis=1)/s)**2)
        if mask is not None:w=w*mask
        if w.sum()<1e-12:return None
        return int(rng.choice(len(P),p=w/w.sum()))
    for _ in range(tries):
        i=pick('index')
        if i is None:continue
        a,na=P[i],N[i];j=pick('thumb',(N@na<-.5)&(np.linalg.norm(P-a,axis=1)<.11))
        if j is None:continue
        k=pick('middle',(N@na>.5)&(np.linalg.norm(P-a,axis=1)>.012)&(np.linalg.norm(P-a,axis=1)<.06))
        if k is None:continue
        b,nb=P[j],N[j];c,nc=P[k],N[k];ps=[self.patch(a,na,0),self.patch(c,nc,1),self.patch(b,nb,2)]
        cov=demand_coverage(ps,{x:8*self.mg for x in range(3)},{x:self.mg for x in range(3)},self.motion,self.o0)
        if cov>=SEED.get('min_cov',.3):return dict(index=a.tolist(),middle=c.tolist(),thumb=b.tolist(),n_index=na.tolist(),n_middle=nc.tolist(),n_thumb=nb.tolist(),coverage=cov)
    return None
def closeness(t,sample):
    """How far a grasp is from the reference, at rest, in the object's frame."""
    d=t.p;d.qpos[:]=t.base_q;d.qpos[t.q_arm]=sample['arm'];d.qpos[t.q_f]=sample['closed'];mujoco.mj_kinematics(t.m,d);tips=SEED['tips'];per={x:round(100*float(np.linalg.norm(t.tip_world(d,x)-tips[x])),2) for x in Y.DIGITS};wp,wr=t.pose(d)
    return dict(tips_cm=per,mean_tip_cm=round(float(np.mean([per[x] for x in ('thumb','index','middle')])),2),palm_cm=round(100*float(np.linalg.norm(wp-SEED['palm_p'])),2),palm_deg=round(float(np.degrees((wr*SEED['palm_r'].inv()).magnitude())),1))
if __name__=='__main__':
    ref=sys.argv[sys.argv.index('--reference')+1];SEED['parquet']=ref;i=sys.argv.index('--reference');argv=sys.argv[:i]+sys.argv[i+2:];sys.argv=argv
    Y.YamTask.propose=propose;Y.__dict__['main']();
    out=Path(sys.argv[sys.argv.index('--output')+1]);ep=sys.argv[sys.argv.index('--episode')+1];side=sys.argv[sys.argv.index('--side')+1]
    fl=[a for a in sys.argv if a=='--onset-floor']
    if fl:Y.ONSET_FLOOR=float(sys.argv[sys.argv.index('--onset-floor')+1])
    t=Y.YamTask(ep,side,int(sys.argv[sys.argv.index('--object')+1]) if '--object' in sys.argv else 0);t.set_stage('real');SEED['tips'],SEED['palm_p'],SEED['palm_r']=reference_tips(t,ref)
    print(json.dumps(dict(reference_tips_to_object_cm={x:round(100*float(np.linalg.norm(t.verts-SEED['tips'][x],axis=1).min()),2) for x in Y.DIGITS})),flush=True)
    res=[]
    for f in sorted(out.glob('pass.json'))+[out/'top.json']:
        if f.is_file():
            s=json.loads(f.read_text());s=s[0] if isinstance(s,list) else s;c=closeness(t,s['sample']);res.append(dict(file=f.name,cm=s.get('mean_cm'),deg=s.get('mean_deg'),frontier=s.get('frontier'),lift_mm=s.get('lift_mm'),**c))
    (out/'closeness.json').write_text(json.dumps(res,indent=1));print(json.dumps(res),flush=True)
