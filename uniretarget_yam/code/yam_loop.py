"""Frontier refinement on YAM arms: every grasp and carry is solved through the arm's joints.

Reachability, joint limits and the 3 rad/s speed limit shape proposals directly, so a pose
that only a floating hand could reach is never tried. Physics on randomized trials decides.
"""
import argparse,json,re,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.contact_wrench import ContactPatch
from flash_chord.retargeting.rigid_wrench_targets import rigid_contact_wrench_targets
from flash_chord.retargeting.surface_contact_search import demand_coverage

FPS,SUBSTEPS=20.,20;TOL_P,TOL_R=.03,np.radians(30.);LOST_P,LOST_R=.15,np.radians(150.);TIP_RADIUS=.007
READY=np.array([0.,1.536,.945,-.61,0.,0.]);ARM_SPEED=3.;DIGITS=('thumb','index','middle','ring','pinky');ROBUST_TRIALS,ROBUST_MINIMUM=24,.9;NATURAL_PALM_MIN=.1;CATCH_UP=int(2.*FPS);PUSH_FRAMES,PUSH_ARM=3,.03;ONSET_FLOOR=2.;SEARCH_PUSH=0.;FINGERS=(3,);VMAX=(.15,.5,.8);WMAX=(.8,3.,4.)  # Caps: initial low and high, mutation ceiling.


def smooth(u):
    u=np.clip(u,0.,1.);return u*u*u*(10-15*u+6*u*u)


def spent_over(ep,er,total):
    # The official metric is an episode mean: excursions are fine while the final mean can pass.
    f=len(ep);limit=.7*f+.3*total;return sum(ep)>TOL_P*limit or sum(er)>TOL_R*limit


class YamTask:
    def __init__(self,episode,side,object_index,future=(),ignored=(),parked=()):
        e=Path(episode);self.m=m=mujoco.MjModel.from_binary_path(str(e/'yam/yam_model.mjb'));self.d=mujoco.MjData(m);self.p=mujoco.MjData(m);self.side=side;self.k=object_index;self.future=set(future);self.ignored=set(ignored)
        with np.load(e/'baseline/physics.npz') as z:self.poses=z['object_pose'][:,0].copy();q0=z['native_qpos'][0,0].copy()
        self.frames=len(self.poses)+1;self.up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);self.jid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
        P=f'{side}_arm_';self.prefix=P;self.arm=[self.jid(f'{P}joint{i}') for i in range(1,7)];self.fingers=[j for j in range(m.njnt) if m.joint(j).name.startswith(f'{P}{side}_') and m.jnt_type[j]==3];self.palm=m.body(f'{P}{side}_hand_C_MC').id
        # Direction halfway between toward-the-other-arm and down, fixed by the two base positions.
        other='left' if side=='right' else 'right';mujoco.mj_kinematics(m,self.p);inward=self.p.xpos[m.body(f'{other}_arm_base_link').id]-self.p.xpos[m.body(f'{P}base_link').id];inward=inward-(inward@self.up)*self.up;inward/=np.linalg.norm(inward);self.natural_palm=(inward-self.up)/np.sqrt(2.)
        self.q_arm=[int(m.jnt_qposadr[j]) for j in self.arm];self.v_arm=[int(m.jnt_dofadr[j]) for j in self.arm];self.q_f=[int(m.jnt_qposadr[j]) for j in self.fingers]
        self.act={int(m.actuator_trnid[a,0]):a for a in range(m.nu)};self.act_arm=[self.act[j] for j in self.arm];self.act_f=[self.act[j] for j in self.fingers]
        self.kd_over_kp=np.array([-m.actuator_biasprm[a,2]/m.actuator_gainprm[a,0] for a in self.act_arm])
        self.free=[j for j in range(m.njnt) if m.jnt_type[j]==0];self.object_q=[]
        for k in range(self.poses.shape[1]):self.object_q.append(min((int(m.jnt_qposadr[j]) for j in self.free),key=lambda q:float(np.linalg.norm(m.qpos0[q:q+3]-q0[self._native_pos_index(q0,k)]))))
        self.object_q=[min((int(m.jnt_qposadr[j]) for j in self.free),key=lambda q:float(np.linalg.norm(m.qpos0[q:q+3]-self.poses[10 if len(self.poses)>10 else 0,k,:3]))) for k in range(self.poses.shape[1])]
        self.qo=self.object_q[self.k];self.obj=int(m.jnt_bodyid[[j for j in self.free if int(m.jnt_qposadr[j])==self.qo][0]]);self.right={b for b in range(m.nbody) if m.body(b).name.startswith(P)}
        # Goal, aligned as the official metric aligns it, and the settled pose grasps are planned on.
        o=self.poses[:,self.k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();self.gp_raw=np.vstack([al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10])]*1);self.gr_raw=al*R.from_quat(o[:,10:14]);self.gp_raw=np.vstack([self.gp_raw,self.gp_raw[-1]]);self.gr_raw=R.concatenate([self.gr_raw,self.gr_raw[-1:]])
        q=(self.gr_raw[0].inv()*self.gr_raw).as_quat()
        for t in range(1,len(q)):
            if q[t]@q[t-1]<0:q[t]=-q[t]
        self.gp=gaussian_filter1d(self.gp_raw,1.5,axis=0,mode='nearest');q=gaussian_filter1d(q,1.5,axis=0,mode='nearest');self.gr=self.gr_raw[0]*R.from_quat(q/np.linalg.norm(q,axis=1,keepdims=True))
        rest=o[min(10,len(o)-1)];self.o0=rest[:3].copy();self.r0=R.from_quat(rest[3:7]);self.reset_pose=np.r_[i[:3],i[3:7][[3,0,1,2]]]
        moved=(np.linalg.norm(self.gp_raw-self.gp_raw[0],axis=1)>.015)|((self.gr_raw*self.gr_raw[0].inv()).magnitude()>.26);self.onset=max(int(np.flatnonzero(moved)[0]) if moved.any() else len(moved)-1,int(ONSET_FLOOR*FPS));self.other_onset={}
        for k in range(self.poses.shape[1]):
            x=self.poses[:,k];a=x[0];b=R.from_quat(a[3:7])*R.from_quat(a[10:14]).inv();g=b.apply(x[:,7:10])+a[:3]-b.apply(a[7:10]);h=b*R.from_quat(x[:,10:14]);mv=(np.linalg.norm(g-g[0],axis=1)>.02)|((h*h[0].inv()).magnitude()>.35);self.other_onset[k]=int(np.flatnonzero(mv)[0]) if mv.any() else len(g)
        self.others=[(k,q_) for k,q_ in enumerate(self.object_q) if k!=self.k];self.stage='real'
        # Fingertips and chains.
        self.tips={};self.chain={}
        for x in DIGITS:
            tip=m.body(f'{P}{side}_{x}_DP').id;gs=[g for g in np.flatnonzero(m.geom_bodyid==tip) if m.geom_contype[g]+m.geom_conaffinity[g]>0];self.tips[x]=(tip,m.geom_pos[gs].mean(0).copy());self.chain[x]=[j for j in self.fingers if f'_{x}_' in m.joint(j).name]
        op={}
        for j in self.fingers:
            lo,hi=m.jnt_range[j];name=m.joint(j).name;op[j]=float(np.clip(.15 if hi>abs(lo)*1.5 else 0.,lo,hi))
            if 'thumb_CMC_FE' in name:op[j]=float(.55*hi)
        self.q_open=np.array([op[j] for j in self.fingers]);self.base_q=m.qpos0.copy()
        for k,q_ in enumerate(self.object_q):x=self.poses[0,k];self.base_q[q_:q_+3]=x[:3];self.base_q[q_+3:q_+7]=x[3:7][[3,0,1,2]]
        for s_ in ('left','right'):
            for i_,v in enumerate(READY,1):self.base_q[m.jnt_qposadr[self.jid(f'{s_}_arm_joint{i_}')]]=v
            for j in range(m.njnt):
                n=m.joint(j).name
                if n.startswith(f'{s_}_arm_{s_}_'):
                    lo,hi=m.jnt_range[j];v=float(np.clip(.15 if hi>abs(lo)*1.5 else 0.,lo,hi));v=float(.55*hi) if 'thumb_CMC_FE' in n else v;self.base_q[m.jnt_qposadr[j]]=v
        self.base_ctrl=np.array([self.base_q[m.jnt_qposadr[m.actuator_trnid[a,0]]] for a in range(m.nu)])
        # Local hand axes at the ready posture.
        d=self.p;d.qpos[:]=self.base_q;mujoco.mj_kinematics(m,d);w=self.pose(d);four=[self.tip_world(d,x) for x in DIGITS[1:]];f=np.mean(four,0)-w[0];f/=np.linalg.norm(f);before=np.mean(four,0);q_=d.qpos.copy();d.qpos[[int(m.jnt_qposadr[j]) for j in self.fingers if 'MCP_FE' in m.joint(j).name and 'thumb' not in m.joint(j).name]]+=.3;mujoco.mj_kinematics(m,d);n=np.mean([self.tip_world(d,x) for x in DIGITS[1:]],0)-before;n-=n@f*f;n/=np.linalg.norm(n);d.qpos[:]=q_;mujoco.mj_kinematics(m,d);self.local_normal=w[1].inv().apply(n)
        # Object surface at the settled pose, for proposals.
        d.qpos[self.qo:self.qo+3]=self.o0;d.qpos[self.qo+3:self.qo+7]=rest[3:7][[3,0,1,2]];mujoco.mj_kinematics(m,d);self.geoms=np.flatnonzero(m.geom_bodyid==self.obj);T=[];N=[];A=[];self.hulls={};V=[];self.rest_geoms={}
        for g in self.geoms:
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].astype(float);h=ConvexHull(v);self.hulls[int(g)]=h.equations;Rg=d.geom_xmat[g].reshape(3,3);pg=d.geom_xpos[g].copy();self.rest_geoms[int(g)]=(pg,Rg.copy());V.append(pg+v@Rg.T)
            for fa,eq in zip(h.simplices,h.equations):
                t=v[fa];a=.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))
                if a>=1e-7:T.append(pg+t@Rg.T);N.append(Rg@eq[:3]);A.append(a)
        self.tri,self.tri_n,self.tri_a=np.asarray(T),np.asarray(N),np.asarray(A);self.verts=np.vstack(V);self.low=float((self.verts@self.up).min());self.mg=float(m.body_mass[self.obj]*np.linalg.norm(m.opt.gravity));self.com=self.o0+self.r0.apply(m.body_ipos[self.obj])
        de=rigid_contact_wrench_targets(self.gp_raw,self.gr_raw.as_quat(),FPS,float(m.body_mass[self.obj]),m.body_inertia[self.obj],m.body_iquat[self.obj][[1,2,3,0]],m.body_ipos[self.obj],m.opt.gravity);fr=np.arange(self.onset,len(de),10);self.motion=np.array([np.r_[(self.r0*self.gr_raw[int(f)].inv()).apply(de[f,:3]),(self.r0*self.gr_raw[int(f)].inv()).apply(de[f,3:])] for f in fr]);self.nominal=(m.geom_friction.copy(),m.body_mass.copy(),m.body_inertia.copy(),float(m.opt.timestep));self.ik_m=mujoco.MjModel.from_binary_path(str(e/'yam/yam_model.mjb'));self.ik_m.geom_margin[:]=.004;self.ik_d=mujoco.MjData(self.ik_m)

    def _native_pos_index(self,q0,k):
        return np.arange(3)

    def pose(self,d):
        return d.xpos[self.palm].copy(),R.from_matrix(d.xmat[self.palm].reshape(3,3).copy())

    def tip_world(self,d,x):
        b,l=self.tips[x];return d.xpos[b]+d.xmat[b].reshape(3,3)@l

    def arm_ik(self,q,p,r,steps=25,limit=None,gain=.7,d=None,m=None):
        """DLS toward a palm pose; limit caps each joint's total change, which is the speed limit."""
        m=m or self.m;d=d or self.p;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);q=np.array(q,float);q0=q.copy();lo,hi=m.jnt_range[self.arm,0],m.jnt_range[self.arm,1]
        for _ in range(steps):
            d.qpos[self.q_arm]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pc=d.xpos[self.palm].copy();rc=R.from_matrix(d.xmat[self.palm].reshape(3,3).copy());e=np.r_[p-pc,.3*(r*rc.inv()).as_rotvec()]
            if np.linalg.norm(e[:3])<4e-4 and np.linalg.norm(e[3:])<.0015:break
            mujoco.mj_jacBody(m,d,jp,jr,self.palm);J=np.r_[jp[:,self.v_arm],.3*jr[:,self.v_arm]];q=q+gain*J.T@np.linalg.solve(J@J.T+1e-3*np.eye(6),e)
            if limit is not None:q=q0+np.clip(q-q0,-limit,limit)
            q=np.clip(q,lo,hi)
        d.qpos[self.q_arm]=q;mujoco.mj_kinematics(m,d);pc,rc=self.pose(d);return q,float(np.linalg.norm(p-pc)),float((r*rc.inv()).magnitude())

    def surface(self,rng,n):
        i=rng.choice(len(self.tri),n,p=self.tri_a/self.tri_a.sum());u=rng.random((n,2));fl=u.sum(1)>1;u[fl]=1-u[fl];t=self.tri[i];p=t[:,0]+u[:,0,None]*(t[:,1]-t[:,0])+u[:,1,None]*(t[:,2]-t[:,0]);nn=self.tri_n[i];keep=np.ones(n,bool)
        for g in self.geoms:
            pg,Rg=self.rest_geoms[int(g)];local=(p+1e-4*nn-pg)@Rg;eq=self.hulls[int(g)];keep&=~(np.max(local@eq[:,:3].T+eq[:,3],axis=1)<0)
        h=p@self.up-self.low;keep&=~((nn@self.up<-.3)&(h<2*TIP_RADIUS+.002));keep&=h>=TIP_RADIUS;return p[keep],nn[keep]

    def patch(self,p,n,g):
        f=-np.asarray(n);ax=np.eye(3)[np.argmin(np.abs(f))];t=np.cross(f,ax);t/=np.linalg.norm(t);return ContactPatch(tuple(p),tuple(f),tuple(t),tuple(np.cross(f,t)),(.7,.7),g)

    def propose(self,rng,tries=30):
        P,N=self.surface(rng,1500)
        for _ in range(tries):
            w=np.exp(-(np.linalg.norm(P-self.com,axis=1)/.05)**2);i=int(rng.choice(len(P),p=w/w.sum()));a,na=P[i],N[i];li=P-a;dist=np.linalg.norm(li,axis=1)+1e-9;ok=(N@na<-.85)&(dist>.008)&(dist<.07)&((li/dist[:,None])@(-na)>.85)
            if not ok.any():continue
            j=int(rng.choice(np.flatnonzero(ok)));b,nb=P[j],N[j];sep=np.linalg.norm(P-a,axis=1);near=(N@na>.7)&(sep>.015)&(sep<.05)
            if not near.any():continue
            k=int(rng.choice(np.flatnonzero(near)));c,nc=P[k],N[k];ps=[self.patch(a,na,0),self.patch(c,nc,1),self.patch(b,nb,2)];extra={};prev,away=c,c-a
            # Ring and pinky continue the line index to middle on the same side; survival in physics decides whether they help.
            for x in ('ring','pinky')[:int(rng.choice(FINGERS))-3]:
                gap=np.linalg.norm(P-prev,axis=1);nextto=(N@na>.7)&(gap>.012)&(gap<.04)&((P-prev)@away>0)
                if not nextto.any():break
                i_=int(rng.choice(np.flatnonzero(nextto)));extra[x]=P[i_].tolist();extra['n_'+x]=N[i_].tolist();ps.append(self.patch(P[i_],N[i_],len(ps)));prev=P[i_]
            cov=demand_coverage(ps,{x:8*self.mg for x in range(len(ps))},{x:self.mg for x in range(len(ps))},self.motion,self.o0)
            if cov>=.5:return dict(index=a.tolist(),middle=c.tolist(),thumb=b.tolist(),n_index=na.tolist(),n_middle=nc.tolist(),n_thumb=nb.tolist(),coverage=cov,**extra)
        return None

    def solve_hand(self,g,rng):
        """Arm and three digits together onto the contacts, from an aligned seed, with collisions pushed apart."""
        m,d=self.ik_m,self.ik_d;a,c,b=(np.asarray(g[x]) for x in('index','middle','thumb'));na,nc,nb=(np.asarray(g[x]) for x in('n_index','n_middle','n_thumb'));goal={'index':a+TIP_RADIUS*na,'middle':c+TIP_RADIUS*nc,'thumb':b+TIP_RADIUS*nb};digits=('thumb','index','middle')+tuple(x for x in('ring','pinky') if x in g);goal.update({x:np.asarray(g[x])+TIP_RADIUS*np.asarray(g['n_'+x]) for x in digits[3:]});d.qpos[:]=self.base_q
        for k,q_ in enumerate(self.object_q):x=self.poses[min(10,len(self.poses)-1),k];d.qpos[q_:q_+3]=x[:3];d.qpos[q_+3:q_+7]=x[3:7][[3,0,1,2]]
        q=np.array(self.q_open);q+=.35*(m.jnt_range[self.fingers,1]-q)*np.array([0 if abs(m.jnt_range[j,0])>.3 and abs(m.jnt_range[j,1])<.5 else 1 for j in self.fingers]);d.qpos[self.q_f]=q;mujoco.mj_kinematics(m,d);ti=self.tip_world(d,'index')-self.tip_world(d,'thumb');wp,wr=self.pose(d);mid=.5*(self.tip_world(d,'index')+self.tip_world(d,'thumb'));normal=wr.apply(self.local_normal);rot,_=R.align_vectors(np.array([goal['index']-goal['thumb'],R.from_rotvec(rng.normal(0,.35,3)).apply(-self.up)]),np.array([ti,normal]),weights=[3.,1.]);tp=.5*(goal['index']+goal['thumb'])+rot.apply(wp-mid);tr=rot*wr
        best=None
        for _ in range(4):
            start=np.clip(READY+rng.normal(0,.5,6),m.jnt_range[self.arm,0],m.jnt_range[self.arm,1]);qa,e1,e2=self.arm_ik(start,tp,tr,steps=40,d=d,m=m)
            if best is None or e1+.05*e2<best[1]+.05*best[2]:best=(qa,e1,e2)
        if best[1]>.04 or best[2]>.7:return None  # The arm cannot bring the hand near this grasp at all.
        d.qpos[self.q_arm]=best[0];dof=self.v_arm+[int(m.jnt_dofadr[j]) for x in digits for j in self.chain[x]];lim=self.arm+[j for x in digits for j in self.chain[x]];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);jq=np.zeros_like(jp);jqr=np.zeros_like(jp);wanted={self.tips[x][0] for x in digits};W=np.ones(len(dof));W[:6]=.25
        for _ in range(70):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);rows=[];err=[]
            for x in digits:
                b_,l=self.tips[x];pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);rows.append(jp[:,dof].copy());err.append(goal[x]-pt)
            for i in range(d.ncon):
                con=d.contact[i];ga,gb=int(m.geom_bodyid[con.geom[0]]),int(m.geom_bodyid[con.geom[1]])
                if not (ga in self.right or gb in self.right):continue
                if {ga,gb}&wanted and self.obj in(ga,gb):continue
                if con.dist>.002:continue
                nrm=con.frame[:3];mujoco.mj_jac(m,d,jp,jr,con.pos,ga);mujoco.mj_jac(m,d,jq,jqr,con.pos,gb);rows.append(((jq-jp)[:,dof]*nrm[:,None]).sum(0,keepdims=True)*3.);err.append(3.*np.array([.002-float(con.dist)]))
            J=np.vstack(rows);e=np.clip(np.concatenate(err),-.015,.015);JW=J*W;dq=W*(JW.T@np.linalg.solve(JW@JW.T+1e-3*np.eye(len(e)),e));full=np.zeros(m.nv);full[dof]=dq;mujoco.mj_integratePos(m,d.qpos,full,1.)
            for j in lim:d.qpos[m.jnt_qposadr[j]]=np.clip(d.qpos[m.jnt_qposadr[j]],*m.jnt_range[j])
        mujoco.mj_kinematics(m,d);residual=max(float(np.linalg.norm(goal[x]-self.tip_world(d,x))) for x in digits);mujoco.mj_collision(m,d);deep=0.
        for i in range(d.ncon):
            con=d.contact[i];x,y=int(m.geom_bodyid[con.geom[0]]),int(m.geom_bodyid[con.geom[1]])
            if x in self.right or y in self.right:
                if {x,y}&wanted and self.obj in(x,y):continue
                deep=max(deep,-float(con.dist))
        wp,wr=self.pose(d)
        if wr.apply(self.local_normal)@self.natural_palm<NATURAL_PALM_MIN:return None  # The palm must face inward or down, never away from the other arm.
        fing=d.qpos[self.q_f].copy();star=d.qpos.copy();pre=star.copy();idx={j:i for i,j in enumerate(self.fingers)}
        for x in digits:
            ch=self.chain[x];dd=[int(m.jnt_dofadr[j]) for j in ch];ad=[int(m.jnt_qposadr[j]) for j in ch];b_,l=self.tips[x];n_=np.asarray(g['n_'+x]);d.qpos[:]=star;mujoco.mj_kinematics(m,d);tg=self.tip_world(d,x)+.02*n_
            for _ in range(20):
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);JJ=jp[:,dd];d.qpos[ad]=np.clip(d.qpos[ad]+JJ.T@np.linalg.solve(JJ@JJ.T+1e-3*np.eye(3),np.clip(tg-pt,-.005,.005)),m.jnt_range[ch,0],m.jnt_range[ch,1])
            pre[ad]=d.qpos[ad]
        d.qpos[:]=star
        return dict(p=wp.tolist(),r=wr.as_rotvec().tolist(),arm=d.qpos[self.q_arm].tolist(),closed=fing.tolist(),pre=pre[self.q_f].tolist(),residual_m=residual,penetration_m=deep,normals={x:g['n_'+x] for x in digits},coverage=g['coverage'],press=float(rng.uniform(.005,.02)),clearance=float(rng.uniform(.02,.05)),vmax=float(rng.uniform(*VMAX[:2])),wmax=float(rng.uniform(*WMAX[:2])))

    def carry_score(self,s):
        """Fraction of the goal path this grasp's rigid carry stays reachable at the arm's speed limit."""
        wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);q=np.asarray(s['arm'],float);hit=0;count=0;step=ARM_SPEED/FPS
        gp,gr=self.task_goal[:2] if hasattr(self,'task_goal') else (self.gp,self.gr)
        for f in range(self.onset,self.frames):
            rel=gr[f]*self.r0.inv();q,a,b=self.arm_ik(q,gp[f]+rel.apply(wp-self.o0),rel*wr,steps=10,limit=step);hit+=a<.03 and b<TOL_R;count+=1
        return hit/max(1,count)

    def follow(self,s,n):
        """One clock along the goal under speed limits, lifted just enough to clear the support."""
        v,w=s['vmax'],s['wmax'];ps=np.tile(self.o0,(n,1));rs=[self.r0]*n;p=self.o0.copy();r=self.r0;dt=1/FPS;start=self.onset-2;tau=start;local=self.r0.inv().apply(self.verts[::max(1,len(self.verts)//200)]-self.o0);floor=float((self.verts@self.up).min())
        for f in range(n):
            if f>=start:
                for k in range(tau+1,min(f,len(self.gp)-1)+1):
                    dp=self.gp[k]-p;dw=(self.gr[k]*r.inv()).as_rotvec();lp,lw=float(np.linalg.norm(dp)),float(np.linalg.norm(dw));u=min(1.,v*dt/max(lp,1e-9),w*dt/max(lw,1e-9))
                    if k>tau+1 and u<1.:break
                    p=p+u*dp;r=R.from_rotvec(u*dw)*r
                    if u<1.:break
                    tau=k
            ps[f]=p;rs[f]=r
        lift=np.array([max(0.,floor+.006-float(((rs[f].apply(local)+ps[f])@self.up).min())) for f in range(n)]);lift[:max(0,start)]=0.
        if lift.any():ps=ps+np.maximum(lift,gaussian_filter1d(lift,2.,mode='nearest'))[:,None]*self.up
        edits=s.get('edits',())
        if edits:
            t=np.arange(n);dp=np.zeros((n,3));dr=np.zeros((n,3))
            for e in edits:
                g=np.exp(-.5*((t-e['frame'])/e['width'])**2);dp+=g[:,None]*np.asarray(e['dp']);dr+=g[:,None]*np.asarray(e['dr'])
            ps=ps+dp;rs=[R.from_rotvec(dr[i])*rs[i] for i in range(n)]
        return ps,rs

    def closure_path(self,s,steps=8):
        m=self.m;d=self.p;pre=np.asarray(s['pre'],float);out=np.tile(pre,(steps+1,1));d.qpos[:]=self.base_q;d.qpos[self.q_arm]=s['arm'];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);idx={j:i for i,j in enumerate(self.fingers)}
        for j in [j for j in self.fingers if not any(j in self.chain[x] for x in s['normals'])]:out[:,idx[j]]=np.linspace(pre[idx[j]],s['closed'][idx[j]],steps+1)
        for x,n_ in s['normals'].items():
            ch=self.chain[x];dd=[int(m.jnt_dofadr[j]) for j in ch];ad=[int(m.jnt_qposadr[j]) for j in ch];b_,l=self.tips[x];d.qpos[self.q_f]=s['closed'];mujoco.mj_kinematics(m,d);end=self.tip_world(d,x)-s.get('press_by',{}).get(x,s['press'])*np.asarray(n_);d.qpos[self.q_f]=pre;mujoco.mj_kinematics(m,d);start=self.tip_world(d,x)
            for i in range(1,steps+1):
                g=start+(i/steps)*(end-start)
                for _ in range(10):
                    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);J=jp[:,dd];d.qpos[ad]=np.clip(d.qpos[ad]+J.T@np.linalg.solve(J@J.T+1e-3*np.eye(3),np.clip(g-pt,-.005,.005)),m.jnt_range[ch,0],m.jnt_range[ch,1])
                for j,a_ in zip(ch,ad):out[i,idx[j]]=d.qpos[a_]
        return out

    def compile(self,s,horizon,base_ctrl=None):
        """Control tape [frame, actuator]: arm joint targets respect the speed limit by construction."""
        m=self.m;n=min(horizon+1,self.frames);C=np.tile(self.base_ctrl if base_ctrl is None else base_ctrl[0],(n,1)) if base_ctrl is None else base_ctrl[:n].copy();wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);way=self.closure_path(s);dt=lambda x:int(round(x*FPS));close_end=max(3,self.onset-dt(.15));close_start=max(2,close_end-dt(.4));arrive=max(1,close_start-dt(.3));step=ARM_SPEED/FPS*.9
        d=self.p;d.qpos[:]=self.base_q;normal=wr.apply(self.local_normal);q_grasp,_,_=self.arm_ik(np.asarray(s['arm']),wp,wr,steps=15);q_pre,e1,_=self.arm_ik(q_grasp,wp-s['clearance']*normal,wr,steps=25);hp,hr=self.follow(s,n);Q=np.zeros((n,6));F=np.tile(self.q_open,(n,1));q=READY.copy();Q[0]=q;miss=[]
        for f in range(1,n):
            if f<arrive:
                u=smooth(f/max(1,arrive));q=READY+u*(q_pre-READY);F[f]=self.q_open+u*(np.asarray(s['pre'])-self.q_open)
            elif f<close_start:
                u=smooth((f-arrive)/max(1,close_start-arrive));q=q_pre+u*(q_grasp-q_pre);F[f]=s['pre']
            else:
                u=smooth((f-close_start)/max(1,close_end-close_start));w=u*(len(way)-1);i=min(int(w),len(way)-2);F[f]=way[i]+(w-i)*(way[i+1]-way[i]);rel=hr[f]*self.r0.inv();q,e1,e2=self.arm_ik(q,hp[f]+rel.apply(wp-self.o0),rel*wr,steps=8,limit=step)
                if f>=self.onset:miss.append((e1,e2))
            Q[f]=q
        self.last_Q=Q;V=np.gradient(Q,1/FPS,axis=0)
        C[:,self.act_arm]=Q+self.kd_over_kp*np.clip(V,-ARM_SPEED,ARM_SPEED);C[:,self.act_f]=F;return C,(np.array(miss) if miss else np.zeros((0,2)))

    def randomize(self,rng):
        m=self.m;f,ma,ine,dt=self.nominal;g=self.geoms;sc=float(rng.uniform(1.,1.25));m.geom_friction[:]=f;m.geom_friction[g,0]=f[g,0]*rng.uniform(.8,1.);m.body_mass[:]=ma;m.body_inertia[:]=ine;m.body_mass[self.obj]=ma[self.obj]*sc;m.body_inertia[self.obj]=ine[self.obj]*sc;m.opt.timestep=dt*float(rng.choice([1.,.8,1.25]));self.substeps=int(round(1/(FPS*m.opt.timestep)))

    def nominal_physics(self):
        m=self.m;f,ma,ine,dt=self.nominal;m.geom_friction[:]=f;m.body_mass[:]=ma;m.body_inertia[:]=ine;m.opt.timestep=dt;self.substeps=SUBSTEPS

    def reach_speed(self,s):
        """Peak joint speed of the fifth-order reach from the ready pose to this grasp's pre-grasp, in the time the onset leaves."""
        dt=lambda x:int(round(x*FPS));close_end=max(3,self.onset-dt(.15));close_start=max(2,close_end-dt(.4));arrive=max(1,close_start-dt(.3));wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);self.p.qpos[:]=self.base_q;q_grasp,_,_=self.arm_ik(np.asarray(s['arm']),wp,wr,steps=15);q_pre,_,_=self.arm_ik(q_grasp,wp-s['clearance']*wr.apply(self.local_normal),wr,steps=25);return float(1.875*np.abs(q_pre-READY).max()*FPS/arrive)

    def sample_pushes(self,rng,level,horizon):
        """Short shoves on the object while it should be held: one during the grasp and lift, one later. Level is in object weights."""
        if level<=0:return []
        def unit():
            v=rng.normal(size=3);return v/np.linalg.norm(v)
        lift_end=min(self.onset+40,horizon);frames=[int(rng.integers(self.onset,max(self.onset+1,lift_end)))]
        if horizon>lift_end:frames.append(int(rng.integers(lift_end,horizon)))
        return [(f,np.r_[level*self.mg*unit(),level*self.mg*PUSH_ARM*unit()]) for f in frames]

    def rollout(self,C,horizon,jitter,record=False):
        m,d=self.m,self.d;mujoco.mj_resetData(m,d);d.qpos[:]=self.base_q;d.qpos[self.qo:self.qo+7]=self.reset_pose;d.qpos[self.qo:self.qo+3]+=jitter;mujoco.mj_forward(m,d);ep=[];er=[];ev=[];lift=0.;held=[];disturbed=False;spent={};qs=[];f6=np.zeros(6);peak=0.;n=min(horizon,len(C));steps=getattr(self,'substeps',SUBSTEPS)
        for f in range(n):
            d.ctrl[:]=C[f];d.xfrc_applied[self.obj]=0.
            for f0,w in getattr(self,'pushes',()):
                if f0<=f<f0+PUSH_FRAMES:d.xfrc_applied[self.obj]=w
            for _ in range(steps):mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all():return dict(frontier=f,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,cause=f,frames=f,exploded=True)
            po=d.qpos[self.qo:self.qo+3].copy();ro=R.from_quat(d.qpos[self.qo+3:self.qo+7][[1,2,3,0]]);goal=self.gp_raw[f];rv=(self.gr_raw[f]*ro.inv()).as_rotvec();e1=float(np.linalg.norm(po-goal));e2=float(np.linalg.norm(rv));ep.append(e1);er.append(e2);ev.append(np.r_[goal-po,rv]);lift=max(lift,float((po-self.o0)@self.up));touch=False
            if record:qs.append(d.qpos.copy())
            if f>=self.onset-8:
                up=0.
                for i in range(d.ncon):
                    c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                    if self.obj in(a,b) and (a in self.right or b in self.right):
                        touch=True;mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3];up+=float((fw if b==self.obj else -fw)@self.up)
                peak=max(peak,up/self.mg)
            held.append(touch)
            if self.stage=='lift' and f>self.onset and (e1>.075 or e2>2.5*TOL_R):break
            if self.stage=='real':
                # An arm catching up after closing is not lost; only the error budget binds until then.
                if f>self.onset and ((f>self.onset+CATCH_UP and (e1>LOST_P or e2>LOST_R)) or spent_over(ep,er,self.frames)):break
                if self.others and f%5==0:
                    for k,qi in self.others:
                        if k in self.ignored or (k in self.future and f>=self.other_onset[k]):continue
                        x=self.poses[min(f,len(self.poses)-1),k];i_=self.poses[0,k];al=R.from_quat(i_[3:7])*R.from_quat(i_[10:14]).inv();g=al.apply(x[7:10])+i_[:3]-al.apply(i_[7:10]);t=(R.from_quat(d.qpos[qi+3:qi+7][[1,2,3,0]]).inv()*(al*R.from_quat(x[10:14]))).magnitude();sp=spent.setdefault(k,[0.,0.]);sp[0]+=5*float(np.linalg.norm(d.qpos[qi:qi+3]-g));sp[1]+=5*float(t);lim=.7*(f+1)+.3*self.frames
                        if sp[0]>TOL_P*lim or sp[1]>TOL_R*lim:disturbed=True
                    if disturbed:break
        ep=np.asarray(ep);er=np.asarray(er);ev=np.asarray(ev);held=np.asarray(held)
        if self.stage=='real':frontier=len(ep)-1 if ((len(ep)-1>self.onset+CATCH_UP and (ep[-1]>LOST_P or er[-1]>LOST_R)) or disturbed or spent_over(ep,er,self.frames)) else len(ep)
        else:over=np.flatnonzero((ep>2*TOL_P)|(er>2*TOL_R));frontier=int(over[0]) if len(over) else len(ep)
        before=np.flatnonzero(held[:frontier+1]);cause=int(before[-1]) if len(before) else max(0,frontier-1);win=np.arange(max(0,cause-12),min(len(ev),frontier+1),3);dense=np.arange(min(self.onset,len(ev)-1),len(ev),5)
        r=dict(frontier=frontier,cause=cause,grip_lost=bool(frontier-cause>2),mean_cm=float(100*ep.mean()),mean_deg=float(np.degrees(er.mean())),lift_mm=lift*1e3,hold=int(held.sum()),peak_up=float(peak),frames=len(ep),error_frames=win.tolist(),error=ev[win].tolist(),dense_frames=dense.tolist(),dense_error=ev[dense].tolist())
        if record:r['qpos']=np.asarray(qs)
        return r

    def set_stage(self,stage):
        if not hasattr(self,'task_goal'):self.task_goal=(self.gp,self.gr,self.gp_raw,self.gr_raw)
        self.stage=stage
        if stage=='real':self.gp,self.gr,self.gp_raw,self.gr_raw=self.task_goal
        else:
            n=len(self.task_goal[0]);u=np.clip((np.arange(n)-self.onset)/FPS,0,1);v=u*u*(3-2*u);self.gp=self.gp_raw=self.o0+np.outer(v*.08,self.up);self.gr=self.gr_raw=R.concatenate([self.r0]*n)

    def edit(self,parent,rng):
        o=dict(parent['sample']);ed=list(o.get('edits',()))
        if parent.get('dense_frames') and rng.random()<.4:
            g=float(rng.uniform(.2,.8));ed+=[dict(frame=int(f),width=3.,dp=(g*np.asarray(e[:3])).tolist(),dr=(g*np.asarray(e[3:])).tolist()) for f,e in zip(parent['dense_frames'],parent['dense_error'])];o['edits']=ed[-400:];return o
        fr=parent.get('error_frames') or [parent.get('cause',self.onset)];er=parent.get('error') or [[0.]*6];i=int(rng.integers(len(fr)));v=np.asarray(er[i]);g=0. if parent.get('grip_lost') and rng.random()<.5 else float(rng.uniform(.2,1.));ed.append(dict(frame=int(fr[i])+int(rng.integers(-4,5)),width=float(rng.uniform(2.,10.)),dp=(g*v[:3]+rng.normal(0,.004,3)).tolist(),dr=(g*v[3:]+rng.normal(0,.05,3)).tolist()));o['edits']=ed[-400:];return o


def mutate(s,rng,scale=.6):
    o=dict(s);o['p']=(np.asarray(s['p'])+rng.normal(0,.004*scale,3)).tolist();o['r']=(R.from_rotvec(rng.normal(0,.06*scale,3))*R.from_rotvec(s['r'])).as_rotvec().tolist();o['closed']=(np.asarray(s['closed'])+rng.normal(0,.05*scale,len(s['closed']))).tolist();o['press']=float(np.clip(s['press']+rng.normal(0,.004*scale),.002,.035));o['press_by']={x:float(np.clip(s.get('press_by',{}).get(x,s['press'])+rng.normal(0,.005*scale),.002,.05)) for x in s['normals']};o['clearance']=float(np.clip(s['clearance']+rng.normal(0,.006*scale),.01,.07));o['vmax']=float(np.clip(s['vmax']*np.exp(rng.normal(0,.2*scale)),.1,VMAX[2]));o['wmax']=float(np.clip(s['wmax']*np.exp(rng.normal(0,.2*scale)),.3,WMAX[2]));return o


T_=None;BASE=None
def _init(a):
    global T_,BASE;T_=YamTask(*a[:6]);BASE=np.load(a[6])['ctrl'] if a[6] else None
def _proposal(seed):
    rng=np.random.default_rng(seed)
    for _ in range(60):
        g=T_.propose(rng)
        if g is None:continue
        s=T_.solve_hand(g,rng)
        if s is not None and s['residual_m']<.008 and s['penetration_m']<.008 and T_.reach_speed(s)<=ARM_SPEED:
            s['carry']=T_.carry_score(s);return s
    return None
def _job(job):
    s,h,stage,seed,*push=job;rng=np.random.default_rng(seed);level=push[0] if push else (SEARCH_PUSH if stage=='real' else 0.)
    if T_.stage!=stage:T_.set_stage(stage)
    try:
        T_.randomize(rng);T_.pushes=T_.sample_pushes(rng,level,h);C,miss=T_.compile(s,h,BASE);r=T_.rollout(C,h,rng.uniform(-.003,.003,3)*np.array([1,1,0]));r['ik_miss_cm']=float(100*miss[:,0].mean()) if len(miss) else 0.;r['ik_miss_deg']=float(np.degrees(miss[:,1].mean())) if len(miss) else 0.
    except Exception as err:r=dict(frontier=-1,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,cause=0,frames=0,failure=repr(err)[:200])
    r['sample']=s;r['rung']=stage;return r
def _child(job):
    e,seed,stage,*grip=job;rng=np.random.default_rng(seed);keep_edit=grip[0] if grip else .7
    if T_.stage!=stage:T_.set_stage(stage)
    c=mutate(e['sample'],rng) if stage!='real' or rng.random()>=keep_edit else T_.edit(e,rng);c.pop('cap',None);return c
def rank(r):
    if r.get('rung')=='lift':return (-min(r['lift_mm'],60.),-r['hold'],-r['frontier'],max(r['mean_cm']/3,r['mean_deg']/30))
    # A candidate that never held the object cannot outrank one that did, whatever its error.
    never_held=r['hold']<5 or r['lift_mm']<20.
    return (never_held,-min(r['frontier'],r['sample'].get('cap',10**6)),-r['hold'],max(r['mean_cm']/3,r['mean_deg']/30),-r['lift_mm'])
STRIP=('error','error_frames','dense_frames','dense_error','qpos')


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--proposals',type=int,default=240);p.add_argument('--population',type=int,default=80);p.add_argument('--generations',type=int,default=120);p.add_argument('--stall',type=int,default=25);p.add_argument('--seed',type=int,default=0);p.add_argument('--future',type=int,nargs='*',default=[]);p.add_argument('--ignored',type=int,nargs='*',default=[]);p.add_argument('--base-tape',default='');p.add_argument('--onset-floor',type=float,default=ONSET_FLOOR);p.add_argument('--search-push',type=float,default=0.);p.add_argument('--wide-caps',action='store_true');p.add_argument('--fingers',type=int,nargs='+',default=[3])
    a=p.parse_args();globals().update(ONSET_FLOOR=a.onset_floor,SEARCH_PUSH=a.search_push,FINGERS=tuple(a.fingers));a.wide_caps and globals().update(VMAX=(.15,1.,1.5),WMAX=(.8,6.,8.));a.output.mkdir(parents=True,exist_ok=True);(a.output/'milestones').mkdir(exist_ok=True);log=(a.output/'generations.jsonl').open('a');rng=np.random.default_rng(a.seed);t0=time.monotonic();args=(a.episode,a.side,a.object,a.future,a.ignored,(),a.base_tape or None);c=a.seed*1000000+1;base=YamTask(*args[:6]);frames=base.frames;base_ctrl=np.load(a.base_tape)['ctrl'] if a.base_tape else None;print(json.dumps(dict(stage='start',frames=frames,onset=base.onset,side=a.side,object=a.object)),flush=True)
    def ids(n):
        nonlocal c;r=range(c,c+n);c+=n;return r
    with Pool(a.workers,_init,(args,)) as pool:
        attempts=a.proposals;pop=[s for s in pool.imap_unordered(_proposal,ids(attempts),chunksize=2) if s is not None]
        scores=[s['carry'] for s in pop];print(json.dumps(dict(stage='proposals',kept=len(pop),of=attempts,seconds=round(time.monotonic()-t0),carry_p50_p90_max=np.percentile(scores,[50,90,100]).round(2).tolist() if scores else [])),flush=True)
        # Physics only for grasps the arm can actually carry: the best third, at least twenty.
        pop=sorted(pop,key=lambda s:-s['carry'])[:max(20,len(pop)//3)]
        if not pop:print(json.dumps(dict(stage='no_reachable_proposal')),flush=True);return
        stage='lift';horizon=min(frames-1,base.onset+40);hist=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(pop,ids(len(pop)))],chunksize=2);best=-1;stalled=0;milestone=-1;best_lift=-1.
        for g in range(a.generations):
            hist.sort(key=rank);hist=hist[:600];top=hist[:12];chk=pool.map(_job,[(r['sample'],horizon,stage,i) for r in top for i in ids(3)],chunksize=1);ver=[]
            for k,r in enumerate(top):
                w=min(chk[k*3:(k+1)*3],key=lambda x:(x['frontier'],x['lift_mm']));w['sample']=r['sample'];w['verified']=True;ver.append(w)
            hist=ver+hist[12:];hist.sort(key=rank);b=hist[0];row=dict(generation=g,stage=stage,horizon=horizon,evaluated=c-a.seed*1000000-1,seconds=round(time.monotonic()-t0),frontier=b['frontier'],cause=b['cause'],grip_lost=b.get('grip_lost'),hold=b['hold'],lift_mm=round(b['lift_mm'],1),cm=round(b['mean_cm'],2),deg=round(b['mean_deg'],1),edits=len(b['sample'].get('edits',())),ik_cm=round(b.get('ik_miss_cm',0),2),ik_deg=round(b.get('ik_miss_deg',0),1),lifting=sum(r['lift_mm']>40 for r in hist),failures=sum('failure' in r for r in hist));print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if stage=='real' and b['frontier']>=milestone+15:
                milestone=b['frontier'];(a.output/'milestones'/f'frontier_{milestone:04d}.json').write_text(json.dumps(dict(row=row,result={k:v for k,v in b.items() if k not in STRIP}),indent=1)+'\n')
            improved=(b['frontier']>best) if stage=='real' else (b['lift_mm']>best_lift+1)
            if improved:best=max(best,b['frontier']);best_lift=max(best_lift,b['lift_mm']);stalled=0
            else:stalled+=1
            if stage=='lift' and (sum(r['lift_mm']>40 for r in hist)>=30 or stalled>=a.stall):
                seeds=[r['sample'] for r in hist if r['lift_mm']>40][:80] or [r['sample'] for r in hist[:40]];print(json.dumps(dict(stage='promoted',seeds=len(seeds),lifting=sum(r['lift_mm']>40 for r in hist))),flush=True);stage='real';horizon=min(frames-1,base.onset+60);best=-1;stalled=0;hist=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(seeds,ids(len(seeds)))],chunksize=2);continue
            ready=[r for r in hist if r['frontier']>=horizon-1 and r.get('verified')]
            if stage=='real' and ready:
                if horizon<frames-1:horizon=min(frames-1,horizon+60);hist=pool.map(_job,[(r['sample'],horizon,stage,i) for r,i in zip(hist[:60],ids(60))],chunksize=2);best=-1;stalled=0;continue
                winner=None
                for r in ready[:3]:
                    tr=pool.map(_job,[(r['sample'],horizon,stage,i) for i in ids(ROBUST_TRIALS)],chunksize=1);rate=float(np.mean([t['frontier']>=horizon-1 and t['mean_cm']<100*TOL_P and t['mean_deg']<np.degrees(TOL_R) for t in tr]));line=dict(stage='robustness',rate=rate,cm=r['mean_cm'],deg=r['mean_deg']);print(json.dumps(line),flush=True);log.write(json.dumps(line)+'\n');log.flush()
                    if rate>=ROBUST_MINIMUM:winner=r;winner['robust_pass_rate']=rate;break
                    r['sample']=dict(r['sample'],cap=horizon-2)
                if winner is not None:
                    hist.remove(winner);hist.insert(0,winner);(a.output/'pass.json').write_text(json.dumps({k:v for k,v in winner.items() if k not in STRIP},indent=1)+'\n');print(json.dumps(dict(stage='yam_passed',robust=winner['robust_pass_rate'],cm=winner['mean_cm'],deg=winner['mean_deg'])),flush=True);break
            if stage=='real' and stalled>=a.stall:print(json.dumps(dict(stage='stalled',frontier=best)),flush=True);break
            el=hist[:max(6,len(hist)//20)];kids=pool.map(_child,[(el[int(rng.integers(len(el)))],i,stage) for i in ids(a.population)],chunksize=4);hist+=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(kids,ids(len(kids)))],chunksize=2)
    hist.sort(key=rank);base.nominal_physics();base.set_stage('real');C,_=base.compile(hist[0]['sample'],frames-1,base_ctrl);np.savez_compressed(a.output/'best_tape.npz',ctrl=C);(a.output/'top.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in STRIP} for r in hist[:20]],indent=1)+'\n');print('failures',[r['failure'] for r in hist if 'failure' in r][:3],flush=True)


if __name__=='__main__':main()
