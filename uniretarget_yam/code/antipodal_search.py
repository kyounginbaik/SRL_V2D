"""Antipodal proposals + hand IK + physics from the real reset.

Geometry proposes which surfaces could oppose and carry the motion; physics chooses.
Works from any object's collider hulls and any fingered hand with thumb/digit chains.
"""
import argparse,json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_search import Task,smoothstep,TOL_P,TOL_R,DIGITS
from flash_chord.retargeting.contact_wrench import ContactPatch
from flash_chord.retargeting.rigid_wrench_targets import rigid_contact_wrench_targets
from flash_chord.retargeting.surface_contact_search import demand_coverage

TIP_RADIUS=.007


def args_model(a):return a[0]


class Antipodal(Task):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.minimum_coverage=.5;m=self.m;d=self.scratch;d.qpos[:]=self.s.native(self.Q[0]);d.qpos[self.qo:self.qo+3]=self.o0;d.qpos[self.qo+3:self.qo+7]=self.r0.as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d)
        pts=[];nrm=[];area=[];geoms=np.flatnonzero(m.geom_bodyid==self.obj);hulls={}
        for g in geoms:
            mid=m.geom_dataid[g];s=m.mesh_vertadr[mid];v=m.mesh_vert[s:s+m.mesh_vertnum[mid]].astype(float);h=ConvexHull(v);hulls[int(g)]=h.equations
        for g in geoms:
            mid=m.geom_dataid[g];s=m.mesh_vertadr[mid];v=m.mesh_vert[s:s+m.mesh_vertnum[mid]].astype(float);h=ConvexHull(v);Rg=d.geom_xmat[g].reshape(3,3);pg=d.geom_xpos[g]
            for f,eq in zip(h.simplices,h.equations):
                t=v[f];A=.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))
                if A<1e-7:continue
                pts.append(pg+t@Rg.T);nrm.append(Rg@eq[:3]);area.append(A)
        self.rest_geoms={int(g):(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for g in geoms};self.tri=np.asarray(pts);self.tri_n=np.asarray(nrm);self.tri_a=np.asarray(area);self.geoms=geoms;self.hulls=hulls;self.low=float((self.verts@self.up).min())
        self.mg=float(m.body_mass[self.obj]*np.linalg.norm(m.opt.gravity))
        demands=rigid_contact_wrench_targets(self.gp_raw,self.gr_raw.as_quat(),self.fps,float(m.body_mass[self.obj]),m.body_inertia[self.obj],m.body_iquat[self.obj][[1,2,3,0]],m.body_ipos[self.obj],m.opt.gravity)
        frames=np.arange(self.onset,min(self.onset+90,len(demands)),5);self.motion=np.array([np.r_[(self.r0*self.gr_raw[int(f)].inv()).apply(demands[f,:3]),(self.r0*self.gr_raw[int(f)].inv()).apply(demands[f,3:])] for f in frames])
        self.hand_geoms=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in {int(m.jnt_bodyid[j]) for j in self.joints} and m.geom_contype[g]+m.geom_conaffinity[g]>0])
        self.right={int(m.jnt_bodyid[j]) for j in self.joints}
        # A separate planning model reports near-contacts; the physics model is untouched.
        self.ik_m=mujoco.MjModel.from_binary_path(str(args_model(a)));self.ik_m.geom_margin[:]=.004;self.ik_d=mujoco.MjData(self.ik_m)
        self.tip_bodies={self.tips[x][0] for x in DIGITS}

    def surface(self,rng,n):
        i=rng.choice(len(self.tri),n,p=self.tri_a/self.tri_a.sum());u=rng.random((n,2));flip=u.sum(1)>1;u[flip]=1-u[flip];t=self.tri[i];p=t[:,0]+u[:,0,None]*(t[:,1]-t[:,0])+u[:,1,None]*(t[:,2]-t[:,0]);nn=self.tri_n[i]
        # Keep points on the real outside (not interior decomposition faces) that a tip can reach.
        keep=np.ones(n,bool)
        for g in self.geoms:
            # Resting poses, not whatever the shared scratch data was last used for.
            pg,Rg=self.rest_geoms[int(g)];local=(p+1e-4*nn-pg)@Rg;eq=self.hulls[int(g)];keep&=~(np.max(local@eq[:,:3].T+eq[:,3],axis=1)<0)
        h=p@self.up-self.low;keep&=~((nn@self.up<-.3)&(h<2*TIP_RADIUS+.002));keep&=h>=TIP_RADIUS
        return p[keep],nn[keep]

    def patch(self,p,n,group):
        f=-np.asarray(n);ax=np.eye(3)[np.argmin(np.abs(f))];t1=np.cross(f,ax);t1/=np.linalg.norm(t1);return ContactPatch(tuple(p),tuple(f),tuple(t1),tuple(np.cross(f,t1)),(.7,.7),group)

    def propose(self,rng,tries=200):
        P,N=self.surface(rng,1500)
        for _ in range(tries):
            # Contacts near the centre of mass need the least torque to carry the object.
            weight=np.exp(-(np.linalg.norm(P-self.com,axis=1)/.05)**2);i=int(rng.choice(len(P),p=weight/weight.sum()));a,na=P[i],N[i]
            line=P-a;dist=np.linalg.norm(line,axis=1)+1e-9;ok=(N@na<-.85)&(dist>.008)&(dist<.07)&((line/dist[:,None])@(-na)>.85)
            if not ok.any():continue
            b,nb=P[int(rng.choice(np.flatnonzero(ok)))],None;j=int(np.flatnonzero((P==b).all(1))[0]);nb=N[j]
            sep=np.linalg.norm(P-a,axis=1);near=(N@na>.7)&(sep>.015)&(sep<.05)
            if not near.any():continue
            k=int(rng.choice(np.flatnonzero(near)));c,nc=P[k],N[k]
            patches=[self.patch(a,na,0),self.patch(c,nc,1),self.patch(b,nb,2)];lim={g:8*self.mg for g in range(3)};mn={g:self.mg for g in range(3)}
            cover=demand_coverage(patches,lim,mn,self.motion,self.o0)
            if cover<self.minimum_coverage:continue
            return dict(index=a.tolist(),middle=c.tolist(),thumb=b.tolist(),n_index=na.tolist(),n_middle=nc.tolist(),n_thumb=nb.tolist(),coverage=cover)
        return None

    def solve_hand(self,g,rng):
        """Rigid hand seed, then DLS on wrist + thumb/index/middle onto tip targets."""
        m=self.ik_m;d=self.ik_d;a,c,b=(np.asarray(g[k]) for k in ('index','middle','thumb'));na,nc,nb=(np.asarray(g[k]) for k in ('n_index','n_middle','n_thumb'))
        goal={'index':a+TIP_RADIUS*na,'middle':c+TIP_RADIUS*nc,'thumb':b+TIP_RADIUS*nb}
        d.qpos[:]=self.s.native(self.Q[0])
        for k,qi in enumerate(self.s.object_qpos):
            rest=self.s.poses[min(10,len(self.s.poses)-1),k];d.qpos[qi:qi+3]=rest[:3];d.qpos[qi+3:qi+7]=rest[3:7][[3,0,1,2]]
        q=np.array(self.q_open);q+=.35*(self.m.jnt_range[self.fingers,1]-q)*np.array([0 if abs(self.m.jnt_range[j,0])>.3 and abs(self.m.jnt_range[j,1])<.5 else 1 for j in self.fingers]);d.qpos[self.finger_q]=q
        d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*self.w0).as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d)
        ti=self.tip_world(d,'index')-self.tip_world(d,'thumb');wrist=d.xpos[self.body].copy();mid=.5*(self.tip_world(d,'index')+self.tip_world(d,'thumb'))
        normal=self.w0.apply(self.local_normal);tilt=R.from_rotvec(rng.normal(0,.35,3))
        rot,_=R.align_vectors(np.array([goal['index']-goal['thumb'],tilt.apply(-self.up)]),np.array([ti,normal]),weights=[3.,1.])
        p=.5*(goal['index']+goal['thumb'])+rot.apply(wrist-mid);r=rot*self.w0
        d.qpos[self.slide_q]=p-self.slide_offset;d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*r).as_quat()[[3,0,1,2]]
        digits=('thumb','index','middle');dof=list(range(int(m.jnt_dofadr[self.slides[0]]),int(m.jnt_dofadr[self.ball])+3))+[int(m.jnt_dofadr[j]) for x in digits for j in self.chain[x]]
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);lim_j=[j for x in digits for j in self.chain[x]];damp=1e-3
        wanted={self.tips[x][0] for x in digits};jq=np.zeros((3,m.nv));jqr=np.zeros_like(jq)
        for it in range(90):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);rows=[];err=[]
            for x in digits:
                b_,l=self.tips[x];pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);rows.append(jp[:,dof].copy());err.append(goal[x]-pt)
            # Separate every undesired near-contact along its normal.
            for i in range(d.ncon):
                con=d.contact[i];ga,gb=int(m.geom_bodyid[con.geom[0]]),int(m.geom_bodyid[con.geom[1]])
                if not (ga in self.right or gb in self.right):continue
                if {ga,gb}&wanted and self.obj in (ga,gb):continue
                if con.dist>.002:continue
                nrm=con.frame[:3];mujoco.mj_jac(m,d,jp,jr,con.pos,ga);mujoco.mj_jac(m,d,jq,jqr,con.pos,gb);rows.append(((jq-jp)[:,dof]*nrm[:,None]).sum(0,keepdims=True)*3.);err.append(3.*np.array([.002-float(con.dist)]))
            J=np.vstack(rows);e=np.clip(np.concatenate(err),-.015,.015);W=np.ones(len(dof));W[:3]=.3;W[3:6]=.3;JW=J*W;dq=W*(JW.T@np.linalg.solve(JW@JW.T+damp*np.eye(len(e)),e));full=np.zeros(m.nv);full[dof]=dq
            mujoco.mj_integratePos(m,d.qpos,full,1.)
            for j in lim_j:d.qpos[m.jnt_qposadr[j]]=np.clip(d.qpos[m.jnt_qposadr[j]],*m.jnt_range[j])
        mujoco.mj_kinematics(m,d);residual=max(float(np.linalg.norm(goal[x]-self.tip_world(d,x))) for x in digits)
        mujoco.mj_collision(m,d);deep=0.
        for i in range(d.ncon):
            con=d.contact[i];x,y=int(m.geom_bodyid[con.geom[0]]),int(m.geom_bodyid[con.geom[1]])
            if x in self.right or y in self.right:deep=max(deep,-float(con.dist))
        q_ik=d.qpos.copy();m=self.m;d=self.scratch;d.qpos[:]=q_ik;mujoco.mj_kinematics(m,d)
        wr=R.from_matrix(d.xmat[self.body].reshape(3,3).copy());wp=d.xpos[self.body].copy();fing=d.qpos[self.finger_q].copy()
        # Pre-closure: back each contact digit off along its surface normal.
        star=d.qpos.copy();pre=d.qpos.copy();index={j:i for i,j in enumerate(self.fingers)}
        for x in digits:
            chain=self.chain[x];dd=[int(m.jnt_dofadr[j]) for j in chain];ad=[int(m.jnt_qposadr[j]) for j in chain];b_,l=self.tips[x];n_={'index':na,'middle':nc,'thumb':nb}[x];d.qpos[:]=star
            target=self.tip_world(d,x)+.02*n_
            for _ in range(20):
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);JJ=jp[:,dd];d.qpos[ad]=np.clip(d.qpos[ad]+JJ.T@np.linalg.solve(JJ@JJ.T+1e-3*np.eye(3),np.clip(target-pt,-.005,.005)),m.jnt_range[chain,0],m.jnt_range[chain,1])
            pre[ad]=d.qpos[ad]
        d.qpos[:]=star
        return dict(p=wp.tolist(),r=wr.as_rotvec().tolist(),closed=fing.tolist(),pre=pre[self.finger_q].tolist(),residual_m=residual,penetration_m=deep,normals={k:g['n_'+k] for k in ('index','middle','thumb')},coverage=g['coverage'],press=float(rng.uniform(.005,.02)),clearance=float(rng.uniform(.02,.05)),vmax=float(rng.uniform(.1,.4)),wmax=float(rng.uniform(.5,3.)))

    def sweep(self,s):
        """Deepest collision along descent and closure, before any physics is spent on them.

        Self-contact and support contact count everywhere; a fingertip may only touch
        the object at the very end of closing, where the grasp wants it.
        """
        m=self.ik_m;d=self.ik_d;wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);normal=wr.apply(self.local_normal);pre=np.asarray(s['pre']);closed=np.asarray(s['closed']);tips={self.tips[x][0] for x in s['normals']}
        d.qpos[:]=self.s.native(self.Q[0])
        for k,qi in enumerate(self.s.object_qpos):
            p0=self.s.poses[min(10,len(self.s.poses)-1),k];d.qpos[qi:qi+3]=p0[:3];d.qpos[qi+3:qi+7]=p0[3:7][[3,0,1,2]]
        d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*wr).as_quat()[[3,0,1,2]];worst=0.
        path=[(wp-c*s['clearance']*normal,pre,False) for c in (1.,.5,0.)]+[(wp,pre+u*(closed-pre),u>.6) for u in (.2,.4,.6,.8,1.)]
        for pos,fingers,touching in path:
            d.qpos[self.slide_q]=pos-self.slide_offset;d.qpos[self.finger_q]=fingers;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
            for i in range(d.ncon):
                c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                if not (a in self.right or b in self.right):continue
                if touching and self.obj in (a,b) and ({a,b}&tips):continue
                worst=max(worst,-float(c.dist))
        return worst

    def press_targets(self,s):
        """Fingertip goals past the surface: contact, not the target, sets the final posture."""
        m=self.m;d=self.scratch;d.qpos[self.slide_q]=np.asarray(s['p'])-self.slide_offset;d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*R.from_rotvec(s['r'])).as_quat()[[3,0,1,2]];d.qpos[self.finger_q]=s['closed'];q=d.qpos.copy();out=np.array(s['closed'],float);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);index={j:i for i,j in enumerate(self.fingers)}
        for x,n_ in s['normals'].items():
            chain=self.chain[x];dd=[int(m.jnt_dofadr[j]) for j in chain];ad=[int(m.jnt_qposadr[j]) for j in chain];b_,l=self.tips[x];d.qpos[:]=q;mujoco.mj_kinematics(m,d);target=self.tip_world(d,x)-s.get('press_by',{}).get(x,s['press'])*np.asarray(n_)
            for _ in range(15):
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);JJ=jp[:,dd];d.qpos[ad]=d.qpos[ad]+JJ.T@np.linalg.solve(JJ@JJ.T+1e-3*np.eye(3),np.clip(target-pt,-.005,.005))
            for j,a_ in zip(chain,ad):out[index[j]]=d.qpos[a_]
        d.qpos[:]=q;return out

    def set_rung(self,rung):
        """'lift' swaps in a plain 8 cm lift; 'real' restores the task goal. Same physics."""
        if not hasattr(self,'task_goal'):self.task_goal=(self.gp,self.gr,self.gp_raw,self.gr_raw)
        if rung=='real':self.gp,self.gr,self.gp_raw,self.gr_raw=self.task_goal
        elif rung=='slow':
            # The task's own orientations at one-third speed: gravity must stay supported throughout.
            gp,gr,gpr,grr=self.task_goal;n=len(gp);f=np.arange(n);k=np.where(f<=self.onset,f,self.onset+(f-self.onset)//3);self.gp,self.gr,self.gp_raw,self.gr_raw=gp[k],gr[k],gpr[k],grr[k]
        else:
            n=len(self.task_goal[0]);u=np.clip((np.arange(n)-self.onset)/(1.*self.fps),0,1);v=u*u*(3-2*u);self.gp=self.gp_raw=self.o0+np.outer(v*.08,self.up);self.gr=self.gr_raw=R.concatenate([self.r0]*n)
            if rung=='shake':
                # After the lift, turn +/-60 deg about each object axis: a hinge cannot follow all three.
                _,_,axes=np.linalg.svd(self.verts-self.center,full_matrices=False);turn=np.zeros((n,3));segment=min(int(self.fps),max(8,(n-2-self.onset-int(self.fps))//6))
                for i,ax in enumerate(axes):
                    start=self.onset+int(self.fps)+2*i*segment;t=np.arange(n)-start;w=np.where((t>=0)&(t<2*segment),np.sin(np.pi*np.clip(t/(2*segment),0,1))**2,0.);turn+=w[:,None]*np.radians(60.)*ax
                self.gr=self.gr_raw=R.from_rotvec(turn)*self.r0
        self.rung=rung

    def limited_follow(self,vmax,wmax,n):
        """Hand-side object path: one clock along the goal, so lag never leaves the path's geometry.

        Speed limits slow progress along the goal rather than limiting translation and
        rotation separately, which let rotation run ahead of the lift. The object is then
        raised just enough to clear its support, spending a little position budget.
        """
        ps=np.tile(self.o0,(n,1));rs=[self.r0]*n;p=self.o0.copy();r=self.r0;dt=1/self.fps;start=self.onset-2;tau=start
        local=self.r0.inv().apply(self.verts[::max(1,len(self.verts)//200)]-self.o0);floor=float((self.verts@self.up).min())
        for f in range(n):
            if f>=start:
                for k in range(tau+1,f+1):
                    dp=self.gp[k]-p;dw=(self.gr[k]*r.inv()).as_rotvec();lp,lw=float(np.linalg.norm(dp)),float(np.linalg.norm(dw));u=min(1.,vmax*dt/max(lp,1e-9),wmax*dt/max(lw,1e-9))
                    if k>tau+1 and u<1.:break
                    p=p+u*dp;r=R.from_rotvec(u*dw)*r
                    if u<1.:break
                    tau=k
            ps[f]=p;rs[f]=r
        lift=np.array([max(0.,floor+.006-float(((rs[f].apply(local)+ps[f])@self.up).min())) for f in range(n)]);lift[:max(0,start)]=0.
        if lift.any():
            lift=np.maximum(lift,gaussian_filter1d(lift,2.,mode='nearest'));ps=ps+lift[:,None]*self.up
        return ps,rs

    def closure_path(self,s,steps=8):
        """Joint waypoints moving each working fingertip straight along its contact normal.

        Interpolating joint angles sweeps tips along arcs, which pass beside thin objects.
        """
        m=self.m;d=self.scratch;pre=np.asarray(s['pre'],float);out=np.tile(pre,(steps+1,1));d.qpos[self.slide_q]=np.asarray(s['p'])-self.slide_offset;d.qpos[self.ball_q:self.ball_q+4]=(self.fixed.inv()*R.from_rotvec(s['r'])).as_quat()[[3,0,1,2]];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);index={j:i for i,j in enumerate(self.fingers)}
        idle=[j for j in self.fingers if not any(j in self.chain[x] for x in s['normals'])]
        for j in idle:out[:,index[j]]=np.linspace(pre[index[j]],s['closed'][index[j]],steps+1)
        for x,n_ in s['normals'].items():
            chain=self.chain[x];dd=[int(m.jnt_dofadr[j]) for j in chain];ad=[int(m.jnt_qposadr[j]) for j in chain];b_,l=self.tips[x];d.qpos[self.finger_q]=s['closed'];mujoco.mj_kinematics(m,d);end=self.tip_world(d,x)-s.get('press_by',{}).get(x,s['press'])*np.asarray(n_);d.qpos[self.finger_q]=pre;mujoco.mj_kinematics(m,d);start=self.tip_world(d,x)
            for i in range(1,steps+1):
                goal=start+(i/steps)*(end-start)
                for _ in range(10):
                    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);pt=d.xpos[b_]+d.xmat[b_].reshape(3,3)@l;mujoco.mj_jac(m,d,jp,jr,pt,b_);J=jp[:,dd];d.qpos[ad]=np.clip(d.qpos[ad]+J.T@np.linalg.solve(J@J.T+1e-3*np.eye(3),np.clip(goal-pt,-.005,.005)),m.jnt_range[chain,0],m.jnt_range[chain,1])
                for j,a_ in zip(chain,ad):out[i,index[j]]=d.qpos[a_]
        return out

    def compile_grasp(self,s,horizon):
        m=self.m;wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);closed=np.asarray(s['closed']);pre_f=np.asarray(s['pre']);way=self.closure_path(s)
        dt=lambda x:int(round(x*self.fps));close_end=max(3,self.onset-dt(.15));close_start=max(2,close_end-dt(.4));arrive=max(1,close_start-dt(.3));leave=1
        normal=wr.apply(self.local_normal);pre=wp-s['clearance']*normal;n=min(horizon+1,self.frames);G=np.tile(self.s.native(self.Q[0]),(n,1));Tn=self.T[:n].copy();Qn=self.Q[:n].copy()
        q_src=self.s.native(self.Q[leave]);self.scratch.qpos[:]=q_src;mujoco.mj_kinematics(m,self.scratch);p_src=self.scratch.xpos[self.body].copy();r_src=R.from_matrix(self.scratch.xmat[self.body].reshape(3,3).copy());f_src=q_src[self.finger_q]
        Fs=np.zeros((n,len(self.fingers)));Tf=np.zeros_like(Fs);hp,hr=self.limited_follow(s.get('vmax',.25),s.get('wmax',1.5),n)
        for f in range(n):
            if f<=leave:P,Ro,Fs[f]=p_src,r_src,f_src;Tf[f]=f_src
            elif f<arrive:
                u=smoothstep((f-leave)/max(1,arrive-leave));P=(1-u)*p_src+u*pre+.08*np.sin(np.pi*u)*self.up;Ro=r_src*R.from_rotvec(u*(r_src.inv()*wr).as_rotvec());Fs[f]=(1-u)*f_src+u*pre_f;Tf[f]=Fs[f]
            elif f<close_start:
                u=smoothstep((f-arrive)/max(1,close_start-arrive));P=(1-u)*pre+u*wp;Ro=wr;Fs[f]=pre_f;Tf[f]=pre_f
            else:
                u=smoothstep((f-close_start)/max(1,close_end-close_start));rel=hr[f]*self.r0.inv()
                P=hp[f]+rel.apply(wp-self.o0);Ro=rel*wr;w=u*(len(way)-1);i=min(int(w),len(way)-2);Tf[f]=way[i]+(w-i)*(way[i+1]-way[i]);Fs[f]=Tf[f]
            G[f,self.slide_q]=P-self.slide_offset;G[f,self.ball_q:self.ball_q+4]=(self.fixed.inv()*Ro).as_quat()[[3,0,1,2]];G[f,self.finger_q]=Fs[f]
        v=np.zeros((n,m.nv))
        for f in range(n):
            a,b=max(0,f-1),min(n-1,f+1);mujoco.mj_differentiatePos(m,v[f],(b-a)/self.fps,G[a],G[b])
        ac=np.gradient(v,1/self.fps,axis=0);nq=self.s.qs;nd=self.s.ds;jm=self.s.jmap
        for f in range(leave,n):
            out=self.inverse.targets(G[f],v[f],ac[f])
            for j in self.slides:Tn[f,nd[jm[j]]]=out.native_joint_position_targets[j];Qn[f,nq[jm[j]]]=G[f,int(m.jnt_qposadr[j])]
            di=nd[jm[self.ball]];Tn[f,di:di+3]=out.native_ball_rotvec_targets[self.ball];qi=nq[jm[self.ball]];Qn[f,qi:qi+4]=G[f,self.ball_q:self.ball_q+4][[1,2,3,0]]
            for k,j in enumerate(self.fingers):Tn[f,nd[jm[j]]]=Tf[f,k];Qn[f,nq[jm[j]]]=Fs[f,k]
        return Qn,Tn

    def score(self,Qn,Tn,horizon):
        s=self.s;m=self.m;s.reset(Qn[0]);ep=[];er=[];lift=0.;hold=0;f6=np.zeros(6);peak=0.
        for f in range(min(horizon,len(Tn))):
            s.step(Tn[f]);d=s.d;po=d.qpos[self.qo:self.qo+3];ro=R.from_quat(d.qpos[self.qo+3:self.qo+7][[1,2,3,0]]);e1=float(np.linalg.norm(po-self.gp_raw[f]));e2=float((ro.inv()*self.gr_raw[f]).magnitude());ep.append(e1);er.append(e2);lift=max(lift,float((po-self.o0)@self.up))
            if not np.isfinite(d.qpos).all():return dict(frontier=f,exploded=True,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,peak_up=0.)
            if f>self.onset-8:
                up=0.
                for i in range(d.ncon):
                    c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                    if self.obj in (a,b) and (a in self.right or b in self.right):
                        mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3];up+=float((fw if b==self.obj else -fw)@self.up)
                peak=max(peak,up/self.mg);hold+=up>.5*self.mg
            if f>self.onset and (e1>2.5*TOL_P or e2>2.5*TOL_R):break
        ep=np.asarray(ep);er=np.asarray(er);over=np.flatnonzero((ep>2*TOL_P)|(er>2*TOL_R));frontier=int(over[0]) if len(over) else len(ep)
        return dict(frontier=frontier,mean_cm=float(100*ep.mean()),mean_deg=float(np.degrees(er.mean())),lift_mm=lift*1e3,hold=int(hold),peak_up=float(peak),frames=len(ep))


T_=None;ARGS=None
def _init(args):
    global T_,ARGS;ARGS=args;T_=Antipodal(*args[:4]);T_.set_rung(args[4])
def _propose(seed):
    rng=np.random.default_rng(seed)
    for _ in range(40):
        g=T_.propose(rng)
        if g is None:continue
        s=T_.solve_hand(g,rng)
        if s['residual_m']<.006 and s['penetration_m']<.006:return s
    return None
def _eval(job):
    s,h=job
    try:Q,Tn=T_.compile_grasp(s,h);r=T_.score(Q,Tn,h)
    except Exception as err:r=dict(frontier=-1,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,peak_up=0.,error=repr(err)[:150])
    r['sample']=s;return r
def rank(r):return (-r['frontier'],-r['hold'],-r['lift_mm'],max(r['mean_cm']/3,r['mean_deg']/30))
def mutate(s,rng,scale):
    o=dict(s);o['p']=(np.asarray(s['p'])+rng.normal(0,.004*scale,3)).tolist();o['r']=(R.from_rotvec(rng.normal(0,.06*scale,3))*R.from_rotvec(s['r'])).as_rotvec().tolist();o['closed']=(np.asarray(s['closed'])+rng.normal(0,.05*scale,len(s['closed']))).tolist();o['press']=float(np.clip(s['press']+rng.normal(0,.004*scale),.002,.035));o['press_by']={x:float(np.clip(s.get('press_by',{}).get(x,s['press'])+rng.normal(0,.005*scale),.002,.05)) for x in s['normals']};o['clearance']=float(np.clip(s['clearance']+rng.normal(0,.006*scale),.01,.07));o['vmax']=float(np.clip(s.get('vmax',.25)*np.exp(rng.normal(0,.2*scale)),.05,.6));o['wmax']=float(np.clip(s.get('wmax',1.5)*np.exp(rng.normal(0,.2*scale)),.2,6.));return o

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--horizon',type=int,default=120);p.add_argument('--proposals',type=int,default=300);p.add_argument('--rounds',type=int,default=5);p.add_argument('--per-round',type=int,default=160);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--side',default='right');p.add_argument('--rung',default='lift');p.add_argument('--seeds',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);t0=time.monotonic()
    with Pool(a.workers,_init,((a.model,a.physics,a.plan,a.side,a.rung),)) as pool:
        props=[r['sample'] for r in json.loads(a.seeds.read_text())] if a.seeds else [s for s in pool.imap_unordered(_propose,range(a.seed*100000,a.seed*100000+a.proposals),chunksize=2) if s is not None]
        print(json.dumps(dict(stage='proposals',kept=len(props),seconds=round(time.monotonic()-t0),residual_mm=np.percentile([1e3*s['residual_m'] for s in props],[50,90]).round(1).tolist() if props else [],penetration_mm=np.percentile([1e3*s['penetration_m'] for s in props],[50,90]).round(1).tolist() if props else [],coverage=np.percentile([s['coverage'] for s in props],[10,50]).round(2).tolist() if props else [])),flush=True)
        hist=pool.map(_eval,[(s,a.horizon) for s in props],chunksize=2)
        for k in range(a.rounds+1):
            hist.sort(key=rank);b=hist[0]
            print(json.dumps(dict(round=k,evaluated=len(hist),seconds=round(time.monotonic()-t0),frontier=b['frontier'],hold=b['hold'],peak_up=round(b['peak_up'],2),lift_mm=round(b['lift_mm'],1),cm=round(b['mean_cm'],2),deg=round(b['mean_deg'],1),lifting_gt5cm=sum(r['lift_mm']>50 for r in hist),holding=sum(r['hold']>5 for r in hist),errors=sum('error' in r for r in hist),frontier_q=np.percentile([r['frontier'] for r in hist],[50,90,99]).tolist())),flush=True)
            if k==a.rounds:break
            el=hist[:max(8,len(hist)//20)];hist+=pool.map(_eval,[(mutate(el[int(rng.integers(len(el)))]['sample'],rng,1-.15*k),a.horizon) for _ in range(a.per_round)],chunksize=2)
    hist.sort(key=rank);(a.output/'top.json').write_text(json.dumps(hist[:40],indent=1)+'\n');print('errors',[r['error'] for r in hist if 'error' in r][:3])
if __name__=='__main__':main()
