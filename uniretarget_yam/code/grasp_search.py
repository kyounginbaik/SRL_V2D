"""Physics-first grasp search: contacts emerge from closing; the object goal judges.

Each sample is only a wrist pose, a digit subset and a closure gain. Fingers curl
toward the object and stop on contact. The whole tape runs from the real reset,
approach included. No contact points, skill labels or per-episode constants.
"""
import argparse,json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from fastsim import FastSim
from flash_chord.retargeting.native_inverse_dynamics import NativeScalarPDInverse

DIGITS=('thumb','index','middle','ring','pinky');SUBSETS=(('thumb','index'),('thumb','index','middle'),('thumb','index','middle','ring'),DIGITS)
TOL_P,TOL_R=.03,np.radians(30.)


def smoothstep(u):
    u=np.clip(u,0.,1.);return u*u*u*(10-15*u+6*u*u)


class Task:
    def __init__(self,model,physics,plan,side='right',object_index=0):
        self.s=s=FastSim(model,physics,object_index=object_index);self.m=m=s.m;self.d=s.d;self.side=side;self.scratch=mujoco.MjData(m)
        with np.load(plan) as z:self.Q=np.asarray(z['joint_q'][0],float);self.T=np.asarray(z['joint_target'][0],float)
        self.frames=len(self.Q);self.fps=s.fps;self.qo=s.qo;self.obj=s.obj
        prefix=side+'_';self.joints=[j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(prefix)]
        self.slides=[j for j in self.joints if m.jnt_type[j]==2];self.ball=[j for j in self.joints if m.jnt_type[j]==1][0];self.fingers=[j for j in self.joints if m.jnt_type[j]==3]
        self.body=int(m.jnt_bodyid[self.ball]);self.slide_q=[int(m.jnt_qposadr[j]) for j in self.slides];self.ball_q=int(m.jnt_qposadr[self.ball]);self.finger_q=[int(m.jnt_qposadr[j]) for j in self.fingers]
        self.inverse=NativeScalarPDInverse(m,ball_drives={self.ball:(s.kp,s.tau)})
        self.up=-m.opt.gravity/np.linalg.norm(m.opt.gravity)
        pose=s.pose;init=pose[0];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();off=init[:3]-al.apply(init[7:10])
        gp=al.apply(pose[:,7:10])+off;gr=al*R.from_quat(pose[:,10:14]);gp=np.vstack([gp,gp[-1]]);gr=R.concatenate([gr,gr[-1:]])
        self.gp_raw,self.gr_raw=gp,gr;q=(gr[0].inv()*gr).as_quat()
        for i in range(1,len(q)):
            if q[i]@q[i-1]<0:q[i]=-q[i]
        self.gp=gaussian_filter1d(gp,1.5,axis=0,mode='nearest');q=gaussian_filter1d(q,1.5,axis=0,mode='nearest');self.gr=gr[0]*R.from_quat(q/np.linalg.norm(q,axis=1,keepdims=True))
        # Plan from where the object physically settles; the reset itself stays the task's.
        rest=pose[min(10,len(pose)-1)];self.o0=rest[:3].copy();self.r0=R.from_quat(rest[3:7]);self.reset_pose=np.r_[init[:3],init[3:7][[3,0,1,2]]]
        moved=(np.linalg.norm(gp-gp[0],axis=1)>.015)|((gr*gr[0].inv()).magnitude()>.26);self.onset=max(int(np.flatnonzero(moved)[0]),int(1.75*self.fps))  # Hands need this long to arrive and close; official means allow the brief lag.
        # Digit chains and tips from the kinematic tree.
        self.tips={};self.chain={}
        for digit in DIGITS:
            tip=[b for b in range(m.nbody) if m.body(b).name.endswith(f'{prefix}{digit}_DP')][0];geoms=np.flatnonzero(m.geom_bodyid==tip);self.tips[digit]=(tip,m.geom_pos[geoms].mean(0).copy())
            self.chain[digit]=[j for j in self.fingers if digit in m.joint(j).name]
        # Open posture from each joint's range: flexion low, abduction neutral, thumb opposed.
        q_open={}
        for j in self.fingers:
            lo,hi=m.jnt_range[j];name=m.joint(j).name;q_open[j]=float(np.clip(.15 if hi>abs(lo)*1.5 else 0.,lo,hi))
            if 'thumb_CMC_FE' in name:q_open[j]=float(.55*hi)
        self.q_open=np.array([q_open[j] for j in self.fingers])
        d=self.scratch;d.qpos[:]=s.native(self.Q[0]);d.qpos[self.finger_q]=self.q_open;mujoco.mj_kinematics(m,d);self.w0=R.from_matrix(d.xmat[self.body].reshape(3,3).copy());self.p0=d.xpos[self.body].copy()
        self.fixed=self.w0*R.from_quat(d.qpos[self.ball_q:self.ball_q+4][[1,2,3,0]]).inv();self.slide_offset=self.p0-d.qpos[self.slide_q]
        four=[self.tip_world(d,x) for x in DIGITS[1:]];forward=np.mean(four,0)-self.p0;forward/=np.linalg.norm(forward)
        before=np.mean(four,0);q=d.qpos.copy();d.qpos[[int(m.jnt_qposadr[j]) for j in self.fingers if 'MCP_FE' in m.joint(j).name and 'thumb' not in m.joint(j).name]]+=.3;mujoco.mj_kinematics(m,d)
        normal=np.mean([self.tip_world(d,x) for x in DIGITS[1:]],0)-before;normal-=normal@forward*forward;normal/=np.linalg.norm(normal);d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        self.local_forward=self.w0.inv().apply(forward);self.local_normal=self.w0.inv().apply(normal)
        # Object principal axis at rest, from collider vertices.
        d.qpos[self.qo:self.qo+3]=self.o0;d.qpos[self.qo+3:self.qo+7]=rest[3:7][[3,0,1,2]];mujoco.mj_kinematics(m,d);v=[]
        for g in np.flatnonzero(m.geom_bodyid==self.obj):
            mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];v.append(d.geom_xpos[g]+m.mesh_vert[a:a+m.mesh_vertnum[mid]]@d.geom_xmat[g].reshape(3,3).T)
        self.verts=np.vstack(v);self.center=self.verts.mean(0);_,_,vt=np.linalg.svd(self.verts-self.center,full_matrices=False);self.axis=vt[0];self.axial=(self.verts-self.center)@self.axis
        self.com=self.o0+self.r0.apply(m.body_ipos[self.obj])

    def tip_world(self,d,digit):
        b,l=self.tips[digit];return d.xpos[b]+d.xmat[b].reshape(3,3)@l

    def closed_fingers(self,wrist_p,wrist_r,digits):
        """Curl chosen digits toward the object section beneath the palm; contact stops them."""
        m,d=self.m,self.scratch;q=np.array(self.q_open);d.qpos[:]=self.s.native(self.Q[0]);d.qpos[self.slide_q]=wrist_p-self.slide_offset;d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*wrist_r).as_quat()[[3,0,1,2]]
        d.qpos[self.finger_q]=q;mujoco.mj_kinematics(m,d)
        normal=wrist_r.apply(self.local_normal);forward=wrist_r.apply(self.local_forward);probe=wrist_p+.08*forward+.05*normal
        s=float(np.clip((probe-self.center)@self.axis,self.axial.min(),self.axial.max()));near=np.abs(self.axial-s)<.012
        section=self.verts[near].mean(0) if near.any() else self.center+s*self.axis
        tips={x:self.tip_world(d,x) for x in digits};mean_tip=np.mean([tips[x] for x in digits if x!='thumb'],0)
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);index={j:i for i,j in enumerate(self.fingers)}
        for digit in digits:
            spread=0. if digit=='thumb' else (tips[digit]-mean_tip)@self.axis*.8;goal=section+spread*self.axis;chain=self.chain[digit]
            dofs=[int(m.jnt_dofadr[j]) for j in chain];adr=[int(m.jnt_qposadr[j]) for j in chain];lo=m.jnt_range[chain,0];hi=m.jnt_range[chain,1];b,l=self.tips[digit]
            for _ in range(24):
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);point=d.xpos[b]+d.xmat[b].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,point,b);J=jp[:,dofs];e=np.clip(goal-point,-.01,.01)
                d.qpos[adr]=np.clip(d.qpos[adr]+J.T@np.linalg.solve(J@J.T+1e-3*np.eye(3),e),lo,hi)
        return d.qpos[self.finger_q].copy()

    def compile(self,sample,horizon):
        """sample: wrist_p/rotvec in the resting world, digit subset index, closure gain, clearance."""
        m=self.m;wp=np.asarray(sample['p']);wr=R.from_rotvec(sample['r']);digits=SUBSETS[sample['subset']]
        closed=self.closed_fingers(wp,wr,digits);closed=self.q_open+sample['gain']*(closed-self.q_open)
        lo,hi=m.jnt_range[self.fingers,0],m.jnt_range[self.fingers,1];closed_geom=np.clip(closed,lo,hi)
        dt=lambda s:int(round(s*self.fps));close_end=max(3,self.onset-dt(.1));close_start=max(2,close_end-dt(.5));arrive=max(1,close_start-dt(.25));leave=1
        normal=wr.apply(self.local_normal);pre=wp-sample['clearance']*normal
        n=min(horizon+1,self.frames);G=np.tile(self.s.native(self.Q[0]),(n,1));Tn=self.T[:n].copy();Qn=self.Q[:n].copy()
        q_src=self.s.native(self.Q[leave]);self.scratch.qpos[:]=q_src;mujoco.mj_kinematics(m,self.scratch);p_src=self.scratch.xpos[self.body].copy();r_src=R.from_matrix(self.scratch.xmat[self.body].reshape(3,3).copy());f_src=q_src[self.finger_q]
        Ps=np.zeros((n,3));Rs=[None]*n;Fs=np.zeros((n,len(self.fingers)));Tf=np.zeros_like(Fs)
        for f in range(n):
            if f<=leave:Ps[f],Rs[f],Fs[f]=p_src,r_src,f_src;Tf[f]=f_src
            elif f<arrive:
                u=smoothstep((f-leave)/max(1,arrive-leave));Ps[f]=(1-u)*p_src+u*pre+.08*np.sin(np.pi*u)*self.up;Rs[f]=r_src*R.from_rotvec(u*(r_src.inv()*wr).as_rotvec());Fs[f]=(1-u)*f_src+u*self.q_open;Tf[f]=Fs[f]
            elif f<close_start:
                u=smoothstep((f-arrive)/max(1,close_start-arrive));Ps[f]=(1-u)*pre+u*wp;Rs[f]=wr;Fs[f]=self.q_open;Tf[f]=self.q_open
            else:
                u=smoothstep((f-close_start)/max(1,close_end-close_start));w=smoothstep((f-(self.onset-2))/max(1,dt(.2)));rel=R.from_rotvec(w*(self.gr[f]*self.r0.inv()).as_rotvec())
                Ps[f]=self.o0+w*(self.gp[f]-self.o0)+rel.apply(wp-self.o0);Rs[f]=rel*wr;Fs[f]=self.q_open+u*(closed_geom-self.q_open);Tf[f]=self.q_open+u*(closed-self.q_open)
            G[f,self.slide_q]=Ps[f]-self.slide_offset;G[f,self.ball_q:self.ball_q+4]=(self.fixed.inv()*Rs[f]).as_quat()[[3,0,1,2]];G[f,self.finger_q]=Fs[f]
        v=np.zeros((n,m.nv))
        for f in range(n):
            a,b=max(0,f-1),min(n-1,f+1);mujoco.mj_differentiatePos(m,v[f],(b-a)/self.fps,G[a],G[b])
        ac=np.gradient(v,1/self.fps,axis=0);nq=self.s.qs;nd=self.s.ds;jm=self.s.jmap
        for f in range(leave,n):
            out=self.inverse.targets(G[f],v[f],ac[f])
            for j in self.slides:Tn[f,nd[jm[j]]]=out.native_joint_position_targets[j]
            di=nd[jm[self.ball]];Tn[f,di:di+3]=out.native_ball_rotvec_targets[self.ball]
            for k,j in enumerate(self.fingers):Tn[f,nd[jm[j]]]=Tf[f,k]
            for j in self.slides:Qn[f,nq[jm[j]]]=G[f,int(m.jnt_qposadr[j])]
            qi=nq[jm[self.ball]];Qn[f,qi:qi+4]=G[f,self.ball_q:self.ball_q+4][[1,2,3,0]]
            for k,j in enumerate(self.fingers):Qn[f,nq[jm[j]]]=Fs[f,k]
        return Qn,Tn

    def rollout(self,Qn,Tn,horizon,jitter=None):
        """Run from reset; stop soon after error exceeds twice tolerance (official allows transients)."""
        s=self.s;m=self.m;o=None
        if jitter is not None:o=np.r_[self.o0+jitter,self.r0.as_quat()[[3,0,1,2]]]
        s.reset(Qn[0],o);ep=[];er=[];force=0.;right=set(int(m.jnt_bodyid[j]) for j in self.joints);lift=0.
        for f in range(min(horizon,len(Tn))):
            s.step(Tn[f]);d=s.d;po=d.qpos[self.qo:self.qo+3];ro=R.from_quat(d.qpos[self.qo+3:self.qo+7][[1,2,3,0]]);e1=float(np.linalg.norm(po-self.gp_raw[f]));e2=float((ro.inv()*self.gr_raw[f]).magnitude());ep.append(e1);er.append(e2);lift=max(lift,float((po-self.o0)@self.up))
            if not np.isfinite(d.qpos).all():return dict(frontier=f,exploded=True,mean_cm=99.,mean_deg=999.,lift_mm=0.,horizon=horizon,frames=len(ep))
            if f>self.onset and (e1>2.5*TOL_P or e2>2.5*TOL_R):break
        ep=np.asarray(ep);er=np.asarray(er);over=np.flatnonzero((ep>2*TOL_P)|(er>2*TOL_R));frontier=int(over[0]) if len(over) else len(ep)
        return dict(frontier=frontier,mean_cm=float(100*ep.mean()),mean_deg=float(np.degrees(er.mean())),lift_mm=lift*1e3,horizon=horizon,frames=len(ep))

    def initial(self,rng,mode):
        if mode=='demo':
            # Human-derived wrist pose near the first goal motion, as a prior only.
            G=self.s.native(self.Q[min(self.onset,self.frames-1)]);d=self.scratch;d.qpos[:]=G;mujoco.mj_kinematics(self.m,d);p=d.xpos[self.body].copy();r=R.from_matrix(d.xmat[self.body].reshape(3,3).copy())
            rel=self.r0*self.gr[self.onset].inv();p=self.o0+rel.apply(p-self.gp[self.onset]);r=rel*r
            p=p+rng.normal(0,.015,3);r=R.from_rotvec(rng.normal(0,.25,3))*r
        else:
            # Palm over the object, digits across its long axis at a sampled section.
            s=rng.uniform(np.percentile(self.axial,5),np.percentile(self.axial,95));yaw=rng.choice([-1.,1.]);across=yaw*np.cross(self.up,self.axis);across/=np.linalg.norm(across)
            down=-self.up;side=np.cross(down,across);M=np.c_[across,side,down]
            des=R.from_matrix(M);local=R.from_matrix(np.c_[self.local_forward,np.cross(self.local_normal,self.local_forward),self.local_normal])
            r=R.from_rotvec(rng.uniform(-.6,.6)*self.up)*R.from_rotvec(rng.uniform(-.6,.6)*side)*R.from_rotvec(rng.uniform(-.5,.5)*across)*des*local.inv()
            section=self.center+s*self.axis;n=r.apply(self.local_normal);fw=r.apply(self.local_forward);p=section-rng.uniform(.05,.11)*n-rng.uniform(.02,.1)*fw
        return dict(p=p.tolist(),r=r.as_rotvec().tolist(),subset=int(rng.integers(len(SUBSETS))),gain=float(rng.uniform(.9,1.4)),clearance=float(rng.uniform(.02,.05)),mode=mode)

    def mutate(self,e,rng,scale):
        r=R.from_rotvec(rng.normal(0,.12*scale,3))*R.from_rotvec(e['r']);sub=e['subset'] if rng.random()>.15 else int(rng.integers(len(SUBSETS)))
        return dict(p=(np.asarray(e['p'])+rng.normal(0,.008*scale,3)).tolist(),r=r.as_rotvec().tolist(),subset=sub,gain=float(np.clip(e['gain']+rng.normal(0,.1*scale),.7,1.8)),clearance=float(np.clip(e['clearance']+rng.normal(0,.008*scale),.01,.07)),mode=e.get('mode','')+'+')


TASK=None
def _init(args):
    global TASK;TASK=Task(*args)
def _eval(job):
    sample,horizon=job
    try:
        Q,T=TASK.compile(sample,horizon);r=TASK.rollout(Q,T,horizon)
    except Exception as err:
        r=dict(frontier=-1,mean_cm=99.,mean_deg=999.,lift_mm=0.,error=repr(err)[:120])
    r['sample']=sample;return r
def rank(r):return (-r['frontier'],max(r['mean_cm']/3,r['mean_deg']/30))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=120);p.add_argument('--initial',type=int,default=400);p.add_argument('--rounds',type=int,default=4);p.add_argument('--per-round',type=int,default=160);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--side',default='right')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);t0=time.monotonic();history=[]
    with Pool(a.workers,_init,((a.model,a.physics,a.plan,a.side),)) as pool:
        task=Task(a.model,a.physics,a.plan,a.side);print(json.dumps(dict(onset=task.onset,local_forward=task.local_forward.round(2).tolist(),local_normal=task.local_normal.round(2).tolist())),flush=True)
        jobs=[(task.initial(rng,'demo' if i%2 else 'across'),a.horizon) for i in range(a.initial)]
        results=pool.map(_eval,jobs,chunksize=4);history+=results
        for k in range(a.rounds+1):
            history.sort(key=rank);best=history[0]
            print(json.dumps(dict(round=k,evaluated=len(history),seconds=round(time.monotonic()-t0),best_frontier=best['frontier'],best_cm=round(best['mean_cm'],2),best_deg=round(best['mean_deg'],1),lift_mm=round(best['lift_mm'],1),subset=best['sample']['subset'],mode=best['sample']['mode'],lifting_gt_5cm=sum(r['lift_mm']>50 for r in history),errors=sum('error' in r for r in history),frontier_q=np.percentile([r['frontier'] for r in history],[50,90,99]).tolist())),flush=True)
            if k==a.rounds:break
            elite=history[:max(8,len(history)//20)];scale=1.-.15*k
            jobs=[(task.mutate(elite[int(rng.integers(len(elite)))]['sample'],rng,scale),a.horizon) for _ in range(a.per_round)]
            history+=pool.map(_eval,jobs,chunksize=4)
    history.sort(key=rank);(a.output/'top.json').write_text(json.dumps(history[:40],indent=1)+'\n');print('errors sample',[r['error'] for r in history if 'error' in r][:3])

if __name__=='__main__':main()
