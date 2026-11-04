"""Simulation-in-loop approach, closure and lift optimization.

Only bounded robot servo commands are optimized. Every trial begins with the
same frozen scene state and the same original clock. Objects remain free,
unactuated, and scored against immutable reconstructed targets. A short-horizon
search result is never a task-success certificate; export is replayed over the
full original horizon with the existing independent physics verifier.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing as mp,os,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from floating_runtime import _freeze,execute_reference
from wrist_reference import continuous_euler_xyz
from evaluate_mp_sr import evaluate_files


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');tmp.replace(path)


def blend(u):
    u=np.clip(u,0.,1.);return u*u*u*(10.-15.*u+6.*u*u)


def rotations_between(a,b,w):
    return a*Rotation.from_rotvec((a.inv()*b).as_rotvec()*np.asarray(w)[...,None])


class PickupExperiment:
    def __init__(self,spec):
        self.spec=spec;self.m=mujoco.MjModel.from_xml_path(spec['scene'])
        self.ref=TR.load_reference(spec['reference'],self.m);self.Q=self.ref['qpos'];self.fps=self.ref['fps']
        self.original=TR.load_reference(spec['original_reference'],self.m)['qpos']
        self.side=spec.get('side','right');self.obj=spec['object_name'];self.rig=H.HandRig.build(self.m,f'robot_{self.side}_',self.side)
        self.wcols=np.array([self.rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);self.fkeys=list(self.rig.finger_qpos)
        self.fcols=np.array(list(self.rig.finger_qpos.values()));self.cols=np.r_[self.wcols,self.fcols]
        self.aids=np.array([self.rig.wrist_act[k] for k in ['x','y','z','rx','ry','rz']]+list(self.rig.finger_act.values()))
        self.jids=self.m.actuator_trnid[self.aids,0];self.limits=self.m.jnt_range[self.jids]
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.lo,self.hi=self.m.actuator_ctrlrange.T
        self.ob=self.m.body(self.obj).id;self.oc=int(self.m.joint(self.obj+'_free').qposadr[0])
        self.objects={self.m.joint(j).name[:-5]:int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt) if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE}
        self.object_bodies={name:self.m.body(name).id for name in self.objects}
        for col in self.objects.values():
            if not np.array_equal(self.Q[:,col:col+7],self.original[:,col:col+7]):raise ValueError('object references changed')
        self.anchor=np.asarray(np.load(spec['anchor_pose'])['qpos'],float)
        self.anchor_p=self.anchor[self.oc:self.oc+3];self.anchor_R=Rotation.from_quat(self.anchor[self.oc+3:self.oc+7],scalar_first=True)
        self.local_p=self.anchor_R.inv().apply(self.anchor[self.wcols[:3]]-self.anchor_p)
        self.local_R=self.anchor_R.inv()*Rotation.from_euler('XYZ',self.anchor[self.wcols[3:]])
        self.closed=self.anchor[self.fcols].copy();self.opened=np.asarray(spec['open_fingers'],float)
        if self.opened.shape!=(22,):raise ValueError('open_fingers must match HandRig22-joint order')
        self.T=len(self.Q);self.times=np.arange(self.T)/self.fps;self.ts=spec['times']
        # [grasp6, hover6, open22, closed22, transport6, timing3]
        self.dim=65
        self.scale=np.r_[[.015]*3,[.12]*3,[.02]*3,[.12]*3,[.16]*22,[.16]*22,[.012]*3,[.08]*3,[.10,.07,.10]]
        self.bound=np.r_[[.065]*3,[.8]*3,[.07]*3,[.8]*3,[.8]*22,[.8]*22,[.06]*3,[.5]*3,[.3,.25,.4]]
        if 'scale' in spec:self.scale=np.asarray(spec['scale'],float)
        if 'bound' in spec:self.bound=np.asarray(spec['bound'],float)
        if self.scale.shape!=(self.dim,) or self.bound.shape!=(self.dim,) or np.any(self.scale<=0) or np.any(self.bound<=0):raise ValueError('scale/bound must be65 positive entries')
        self.start_frame=round(self.ts['start']*self.fps);self.start_step=round(self.start_frame/self.fps/self.m.opt.timestep)
        self.last_frame=min(self.T-1,round(spec.get('horizon_s',4.)*self.fps));self.end_step=round(self.last_frame/self.fps/self.m.opt.timestep)
        self.dt=float(self.m.opt.timestep);self.steps_per_frame=round(1./self.fps/self.dt)
        if abs(self.steps_per_frame*self.dt-1/self.fps)>1e-10:raise ValueError('integer physics steps per original frame required')
        self.body_finger={}
        for body in self.rig.body_ids:
            for finger in H.FINGERS:
                if f'_{finger}_' in self.m.body(body).name:self.body_finger[body]=finger
        self.fsites=np.array([self.rig.fingertip_sites[f] for f in ['thumb','index','middle','ring','pinky']])
        self.geom_ids=[g for g in range(self.m.ngeom) if self.m.geom_bodyid[g]==self.ob and self.m.geom_contype[g]]
        # Surface vertices give a dense geometric signal before any contact.
        fk=mujoco.MjData(self.m);fk.qpos[:]=self.anchor;mujoco.mj_forward(self.m,fk);verts=[]
        for g in self.geom_ids:
            mid=self.m.geom_dataid[g]
            if mid<0:continue
            v=self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]].astype(float)
            v=v@fk.geom_xmat[g].reshape(3,3).T+fk.geom_xpos[g]
            verts.extend(self.anchor_R.inv().apply(v-self.anchor_p))
        self.surface=np.asarray(verts);self.tree=cKDTree(self.surface)
        self.up=-np.asarray(self.m.opt.gravity,float)
        self.up/=np.linalg.norm(self.up)
        self.rest_height=float(self.anchor_p@self.up)
        self.prefix=mujoco.MjData(self.m)
        if self.m.nkey:mujoco.mj_resetDataKeyframe(self.m,self.prefix,0)
        mujoco.mj_forward(self.m,self.prefix)
        for step in range(self.start_step):
            self.prefix.ctrl[:]=self.control(self.Q,step)
            mujoco.mj_step(self.m,self.prefix)

    def control(self,Q,step):
        frame=step//self.steps_per_frame;alpha=(step%self.steps_per_frame+1)/self.steps_per_frame
        return np.clip(Q[frame,self.qadr]+alpha*(Q[frame+1,self.qadr]-Q[frame,self.qadr]),self.lo,self.hi)

    def commands(self,x):
        x=np.asarray(x);Q=self.Q.copy();t=self.times
        gp=self.local_p+x[:3];gr=self.local_R*Rotation.from_rotvec(x[3:6])
        grasp_p=self.anchor_p+self.anchor_R.apply(gp);grasp_R=self.anchor_R*gr
        hover_p=grasp_p+np.asarray(self.spec.get('hover_offset',[0.,0.,.10]))+x[6:9]
        hover_R=grasp_R*Rotation.from_rotvec(x[9:12])
        opened=np.clip(self.opened+x[12:34],self.limits[6:,0],self.limits[6:,1])
        closed=np.clip(self.closed+x[34:56],self.limits[6:,0],self.limits[6:,1])
        contact=max(self.ts['hover']+.12,self.ts['contact']+x[62])
        close=contact+max(.10,self.ts['closed']-self.ts['contact']+x[63])
        transport=close+max(.25,self.ts['transport']-self.ts['closed']+x[64])
        start=self.start_frame/self.fps;hover=self.ts['hover'];release=self.ts['release']
        p=Q[:,self.wcols[:3]].copy();rot=Rotation.from_euler('XYZ',Q[:,self.wcols[3:]])
        mats=rot.as_matrix();f=Q[:,self.fcols].copy();initial=Q[self.start_frame]
        initialR=Rotation.from_euler('XYZ',initial[self.wcols[3:]])
        for a,b,pa,pb,ra,rb,fa,fb in [
            (start,hover,initial[self.wcols[:3]],hover_p,initialR,hover_R,initial[self.fcols],opened),
            (hover,contact,hover_p,grasp_p,hover_R,grasp_R,opened,opened),
            (contact,close,grasp_p,grasp_p,grasp_R,grasp_R,opened,closed)]:
            mask=(t>a)&(t<=b);w=blend((t[mask]-a)/(b-a))
            p[mask]=(1-w[:,None])*pa+w[:,None]*pb;mats[mask]=rotations_between(ra,rb,w).as_matrix();f[mask]=(1-w[:,None])*fa+w[:,None]*fb
        active=(t>close)&(t<release+.35)
        objR=Rotation.from_quat(self.original[active,self.oc+3:self.oc+7],scalar_first=True)
        targetp=self.original[active,self.oc:self.oc+3]+objR.apply(np.tile(gp+x[56:59],(active.sum(),1)))
        targetR=objR*gr*Rotation.from_rotvec(x[59:62])
        w=blend((t[active]-close)/(transport-close))
        p[active]=(1-w[:,None])*grasp_p+w[:,None]*targetp
        mats[active]=(grasp_R*Rotation.from_rotvec((grasp_R.inv()*targetR).as_rotvec()*w[:,None])).as_matrix();f[active]=closed
        rel=(t>=release)&(t<release+.35);w=blend((t[rel]-release)/.35)
        p[rel]=(1-w[:,None])*p[rel]+w[:,None]*Q[rel][:,self.wcols[:3]]
        oldR=Rotation.from_matrix(mats[rel]);baseR=rot[rel]
        mats[rel]=(oldR*Rotation.from_rotvec((oldR.inv()*baseR).as_rotvec()*w[:,None])).as_matrix()
        f[rel]=(1-w[:,None])*f[rel]+w[:,None]*Q[rel][:,self.fcols]
        Q[:,self.wcols[:3]]=p;Q[:,self.wcols[3:]]=continuous_euler_xyz(Rotation.from_matrix(mats),self.Q[0,self.wcols[3:]])
        Q[:,self.fcols]=f;Q[:self.start_frame+1]=self.Q[:self.start_frame+1]
        Q[:,self.cols]=np.clip(Q[:,self.cols],self.limits[:,0],self.limits[:,1])
        Q[0]=self.Q[0]
        return Q,dict(contact=contact,closed=close,transport=transport)

    def evaluate(self,x,return_trace=False):
        Q,timing=self.commands(x);d=mujoco.MjData(self.m);mujoco.mj_copyData(d,self.m,self.prefix)
        force=np.zeros(6);rows=[];unstable=False
        for step in range(self.start_step,self.end_step):
            d.ctrl[:]=self.control(Q,step);mujoco.mj_step(self.m,d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:unstable=True;break
            if step%5!=4:continue
            sec=float(d.time);frame=min(self.T-1,round(sec*self.fps));objp=d.qpos[self.oc:self.oc+3]
            objR=Rotation.from_quat(d.qpos[self.oc+3:self.oc+7],scalar_first=True)
            tip_local=(d.site_xpos[self.fsites]-d.xpos[self.ob])@d.xmat[self.ob].reshape(3,3)
            gaps=self.tree.query(tip_local)[0];handforces={};patches=[];tableforce=0.;objtable=False;selfpen=0.;handpen=0.
            for i,c in enumerate(d.contact):
                b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2]);r1=b1 in self.rig.body_ids;r2=b2 in self.rig.body_ids
                if r1 and r2:selfpen=max(selfpen,-float(c.dist))
                if r1!=r2:
                    handpen=max(handpen,-float(c.dist));other=b2 if r1 else b1
                    if self.m.body(other).name=='table':mujoco.mj_contactForce(self.m,d,i,force);tableforce+=max(0.,force[0])
                support=b2 if b1==self.ob else b1 if b2==self.ob else -1
                if support>=0 and not self.m.body(support).name.startswith('robot_'):
                    mujoco.mj_contactForce(self.m,d,i,force)
                    if force[0]>.02:objtable=True
                hand=b2 if b1==self.ob else b1 if b2==self.ob else -1;finger=self.body_finger.get(hand)
                if finger is None:continue
                mujoco.mj_contactForce(self.m,d,i,force);normalforce=max(0.,float(force[0]));handforces[finger]=handforces.get(finger,0.)+normalforce
                if normalforce>.1:patches.append((finger,np.array(c.pos),np.array(c.frame[:3])*(1 if b1==self.ob else -1),float(c.friction[0])))
            qualified=False
            for fa,pa,na,ma in patches:
                if fa!='thumb':continue
                for fb,pb,nb,mb in patches:
                    if fb=='thumb':continue
                    vec=pa-pb;length=np.linalg.norm(vec)
                    if length<1e-5:continue
                    line=vec/length;threshold=1/np.sqrt(1+min(ma,mb)**2)
                    if na@nb<-.5 and na@line>=threshold and nb@(-line)>=threshold:qualified=True
            both=handforces.get('thumb',0.)>.1 and any(v>.1 for f,v in handforces.items() if f!='thumb')
            pos=np.linalg.norm(objp-self.original[frame,self.oc:self.oc+3]);angle=(objR.inv()*Rotation.from_quat(self.original[frame,self.oc+3:self.oc+7],scalar_first=True)).magnitude()
            otherpos=max([np.linalg.norm(d.qpos[a:a+3]-self.original[frame,a:a+3]) for name,a in self.objects.items() if name!=self.obj],default=0.)
            wrist_error=np.linalg.norm(d.qpos[self.wcols[:3]]-Q[frame,self.wcols[:3]])
            rows.append([sec,float(gaps[0]),float(gaps[1:].min()),float(both),float(qualified),float(not objtable),float(objp@self.up-self.rest_height),float(pos),float(angle),float(tableforce),float(wrist_error),float(otherpos),float(selfpen),float(handpen)])
        if unstable or not rows:return dict(cost=1e6,unstable=True)
        a=np.asarray(rows);near=(a[:,0]>=timing['contact']-.25)&(a[:,0]<=timing['closed']+.2)
        # Task scoring windows stay fixed across candidates. Moving the robot
        # closure/transport timing cannot move difficult target frames out of
        # the objective or retime the demonstration.
        hold=a[:,0]>=self.spec.get('score_hold_start_s',self.ts['closed']);track=a[:,0]>=min(self.spec.get('score_track_start_s',self.ts['transport']),self.spec.get('horizon_s',4.)-.3)
        if not np.any(near) or not np.any(hold) or not np.any(track):return dict(cost=1e6,unstable=False)
        unsupported=(a[:,5]>.5)&(a[:,6]>.02)&(a[:,3]>.5)
        qualified_unsupported=unsupported&(a[:,4]>.5)
        gap=np.mean(np.minimum((a[near,1]+a[near,2])/.03,4.))
        pos=np.mean(np.minimum(a[track,7]/.03,8.));ang=np.mean(np.minimum(a[track,8]/.5236,4.))
        cost=2*gap+5*pos+1.5*ang+2*(1-a[hold,3].mean())+2*(1-a[hold,4].mean())+self.spec.get('unsupported_weight',6.)*(1-unsupported[hold].mean())
        cost+=.3*np.minimum(a[:,9]/50.,4.).mean()+.5*np.minimum(a[:,10]/.03,4.).mean()
        cost+=5*np.maximum(a[:,11]-.025,0).mean()/.03+20*np.maximum(a[:,12:14]-.005,0).mean()/.005
        cost+=.002*np.mean((np.asarray(x)/self.bound)**2)
        def longest(mask):
            edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            return float(np.max(ends-starts,initial=0)*self.dt*5)
        result=dict(cost=float(cost),unstable=False,mean_tracking_position_cm=float(a[track,7].mean()*100),mean_tracking_rotation_deg=float(np.rad2deg(a[track,8].mean())),
            both_contact_fraction=float(a[hold,3].mean()),qualified_fraction=float(a[hold,4].mean()),unsupported_contact_fraction=float(unsupported[hold].mean()),
            longest_unsupported_contact_s=longest(unsupported),longest_qualified_unsupported_contact_s=longest(qualified_unsupported),
            near_tip_gap_cm=float(gap*3),max_table_force_n=float(a[:,9].max()),mean_other_object_error_cm=float(a[:,11].mean()*100),timing=timing)
        if return_trace:result['trace']=a
        return result


WORKER=None
def initialize(spec):
    global WORKER;WORKER=PickupExperiment(spec)
def evaluate(x):return WORKER.evaluate(x)


def optimize(spec,output,samples=128,iterations=20,workers=16,seed=44,resume=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);spec=dict(spec)
    if (out/'result.json').exists():raise FileExistsError('choose a fresh experiment output')
    (out/'search_physics').mkdir(parents=True,exist_ok=True)
    _,frozen=_freeze(spec['scene'],out/'search_physics');spec['scene']=str(frozen.resolve())
    exp=PickupExperiment(spec);rng=np.random.default_rng(seed);mean=np.zeros(exp.dim);std=exp.scale.copy();best=mean.copy();best_report=exp.evaluate(best);history=[]
    if resume:
        old=json.loads(Path(resume).read_text());best=np.asarray(old['best_parameters']);mean=best.copy();best_report=exp.evaluate(best)
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,dimension=exp.dim,scale=exp.scale,bound=exp.bound))
    dump(out/'initial.json',best_report);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            noise=rng.normal(size=(samples-2,exp.dim));proposals=np.vstack([best,mean,mean+noise*std]);proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(evaluate,proposals,chunksize=1));order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<best_report['cost']:best=proposals[order[0]].copy();best_report=reports[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.15*exp.scale)
            row=dict(iteration=iteration,best=best_report,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best,mean=mean,std=std));dump(out/f'iteration_{iteration:03d}.json',dict(proposals=proposals,reports=reports))
            print(json.dumps(row),flush=True)
            if best_report['longest_qualified_unsupported_contact_s']>.8 and best_report['mean_tracking_position_cm']<2 and best_report['mean_tracking_rotation_deg']<20:break
    Q,timing=exp.commands(best);refpath=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Physical pickup approach/closure/lift CEM; immutable source object targets/clock; robot commands only',arms_coupled=False,floating_runtime=True)
    short=exp.evaluate(best,return_trace=True);trace=short.pop('trace');np.savez_compressed(out/'search_trace.npz',trace=trace,
        columns=np.array(['time','thumb_gap','other_gap','both','qualified','no_nonrobot_support','height','position_error','rotation_error','table_force','wrist_error','other_error','self_pen','hand_pen']))
    execution=execute_reference(spec['scene'],refpath,out/'execution')
    metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    result=dict(best=best_report,parameters=best,reference=str(refpath),execution=execution,metrics=metrics,history=history,spec=spec)
    dump(out/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=128);p.add_argument('--iterations',type=int,default=20);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=44);p.add_argument('--resume',type=Path)
    args=vars(p.parse_args());args['spec']=json.loads(args['spec'].read_text());optimize(**args)
