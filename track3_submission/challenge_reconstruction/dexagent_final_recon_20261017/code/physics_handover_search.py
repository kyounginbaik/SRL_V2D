"""Actual bilateral handover optimization with a preserved pickup prefix.

This optimizes robot commands, not object motion: the left hand must retain
the spoon after the right hand has physically disengaged. A successful short
handover is followed by full-source-horizon execution and independent replay.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search import PickupExperiment,B,H,TR,blend,dump,_freeze,execute_reference,evaluate_files
from wrist_reference import continuous_euler_xyz

class HandoverExperiment(PickupExperiment):
    def __init__(self,spec):
        super().__init__(spec)
        self.right=H.HandRig.build(self.m,'robot_right_','right')
        self.rc=np.array([self.right.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]+list(self.right.finger_qpos.values()))
        self.ra=np.array([self.right.wrist_act[k] for k in ['x','y','z','rx','ry','rz']]+list(self.right.finger_act.values()))
        self.rlimits=self.m.jnt_range[self.m.actuator_trnid[self.ra,0]]
        self.rrot=Rotation.from_euler('XYZ',self.Q[:,self.rc[3:6]])
        self.dim=60
        # Two full28-DOF hand corrections plus three acquisition timings and
        # one right-release timing. Every actuator stays bounded as supplied.
        self.scale=np.r_[[.004]*3,[.06]*3,[.12]*22,[.0025]*3,[.035]*3,[.07]*22,[.07,.05,.06],.07]
        self.bound=np.r_[[.06]*3,[.7]*3,[.9]*22,[.035]*3,[.4]*3,[.6]*22,[.25,.2,.25],.25]
        self.leftbody=set(self.rig.body_ids);self.rightbody=set(self.right.body_ids)
        self.allfinger_cols=[];self.allfinger_ranges=[]
        for rig in [self.rig,self.right]:
            acts=np.array(list(rig.finger_act.values()));j=self.m.actuator_trnid[acts,0]
            self.allfinger_cols.extend(list(rig.finger_qpos.values()));self.allfinger_ranges.extend(self.m.jnt_range[j])
        self.allfinger_cols=np.array(self.allfinger_cols);self.allfinger_ranges=np.array(self.allfinger_ranges)
        assert len(self.allfinger_cols)==44

    def commands(self,x):
        x=np.asarray(x);left=np.zeros(65);left[:6]=x[:6];left[34:56]=x[6:28];left[12:34]=x[6:28]*.3;left[62:65]=x[56:59]
        Q,timing=super().commands(left)
        start=self.ts['start'];release=self.spec.get('original_right_release_start_s',10.1)+x[59]
        end=release+(self.spec.get('original_right_release_end_s',10.65)-self.spec.get('original_right_release_start_s',10.1))
        w=blend((self.times-start)/.3);y=x[28:56]
        p=self.Q[:,self.rc[:3]]+self.rrot.apply(w[:,None]*y[:3])
        R=self.rrot*Rotation.from_rotvec(w[:,None]*y[3:6]);f=np.clip(self.Q[:,self.rc[6:]]+w[:,None]*y[6:],self.rlimits[6:,0],self.rlimits[6:,1])
        # Release to the actual source withdrawal, not a competing persistent
        # right-hand object-following grasp. The left hand takes over transport.
        a=blend((self.times-release)/(end-release));p=(1-a[:,None])*p+a[:,None]*self.original[:,self.rc[:3]]
        RR=Rotation.from_euler('XYZ',self.original[:,self.rc[3:6]])
        R=R*Rotation.from_rotvec((R.inv()*RR).as_rotvec()*a[:,None])
        f=(1-a[:,None])*f+a[:,None]*self.original[:,self.rc[6:]]
        Q[:,self.rc[:3]]=p;Q[:,self.rc[3:6]]=continuous_euler_xyz(R,self.Q[0,self.rc[3:6]]);Q[:,self.rc[6:]]=f
        Q[:self.start_frame+1]=self.Q[:self.start_frame+1]
        for cols in [self.cols,self.rc]:
            aids=[self.m.actuator(i).id for i in range(self.m.nu) if int(self.qadr[i]) in set(cols)]
            for aid in aids:Q[:,self.qadr[aid]]=np.clip(Q[:,self.qadr[aid]],self.lo[aid],self.hi[aid])
        timing.update(right_release_start=release,right_release_end=end)
        return Q,timing

    def evaluate(self,x,return_trace=False):
        Q,timing=self.commands(x);d=mujoco.MjData(self.m);mujoco.mj_copyData(d,self.m,self.prefix)
        force=np.zeros(6);rows=[]
        for step in range(self.start_step,self.end_step):
            d.ctrl[:]=self.control(Q,step);mujoco.mj_step(self.m,d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:return dict(cost=1e9,unstable=True)
            if step%5!=4:continue
            t=float(d.time);frame=min(self.T-1,round(t*self.fps));obp=d.qpos[self.oc:self.oc+3];obR=Rotation.from_quat(d.qpos[self.oc+3:self.oc+7],scalar_first=True)
            local=(d.site_xpos[self.fsites]-d.xpos[self.ob])@d.xmat[self.ob].reshape(3,3);gaps=self.tree.query(local)[0]
            fingers={};patches=[];rightforce=0.;support=0.;table=0.;selfpen=0.;handpen=0.
            for i,c in enumerate(d.contact):
                b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2]);h1=b1 in self.leftbody|self.rightbody;h2=b2 in self.leftbody|self.rightbody
                if h1 and h2:selfpen=max(selfpen,-float(c.dist))
                if h1!=h2:
                    handpen=max(handpen,-float(c.dist));other=b2 if h1 else b1
                    if self.m.body(other).name=='table':mujoco.mj_contactForce(self.m,d,i,force);table+=max(0.,force[0])
                other=b2 if b1==self.ob else b1 if b2==self.ob else -1
                if other<0:continue
                mujoco.mj_contactForce(self.m,d,i,force);fn=max(0.,float(force[0]))
                if other in self.rightbody:rightforce+=fn
                elif other not in self.leftbody:support+=fn
                finger=self.body_finger.get(other)
                if finger is None:continue
                fingers[finger]=fingers.get(finger,0.)+fn
                if fn>.1:patches.append((finger,np.array(c.pos),np.array(c.frame[:3])*(1 if b1==self.ob else -1),float(c.friction[0])))
            qualified=False
            for fa,pa,na,ma in patches:
                if fa!='thumb':continue
                for fb,pb,nb,mb in patches:
                    if fb=='thumb':continue
                    delta=pa-pb;norm=np.linalg.norm(delta)
                    if norm<1e-5:continue
                    line=delta/norm;threshold=1/np.sqrt(1+min(ma,mb)**2)
                    if na@nb<-.5 and na@line>=threshold and nb@(-line)>=threshold:qualified=True
            both=fingers.get('thumb',0)>.1 and any(v>.1 for f,v in fingers.items() if f!='thumb')
            p=np.linalg.norm(obp-self.original[frame,self.oc:self.oc+3]);r=(obR.inv()*Rotation.from_quat(self.original[frame,self.oc+3:self.oc+7],scalar_first=True)).magnitude()
            q=d.qpos[self.allfinger_cols];violation=np.maximum(np.maximum(self.allfinger_ranges[:,0]-q,q-self.allfinger_ranges[:,1]),0).max()
            rows.append([t,gaps[0],gaps[1:].min(),both,qualified,rightforce,support,p,r,table,selfpen,handpen,violation])
        a=np.array(rows);near=(a[:,0]>=9.5)&(a[:,0]<=10.4);hold=a[:,0]>=10.7
        left_alone=(a[:,3]>.5)&(a[:,5]<.1)&(a[:,6]<.1);qualified=left_alone&(a[:,4]>.5)
        pos=np.minimum(a[hold,7]/.03,8).mean();ang=np.minimum(a[hold,8]/.5236,4).mean()
        cost=5*pos+2*ang+2*np.minimum((a[near,1]+a[near,2])/.03,4).mean()+5*(1-left_alone[hold].mean())+3*(1-qualified[hold].mean())
        cost+=2*np.minimum(a[hold,5]/5.,4.).mean()+.2*np.minimum(a[:,9]/30.,4.).mean()
        cost+=10*np.maximum(a[:,10]-.002,0).mean()/.002+10*np.maximum(a[:,11]-.003,0).mean()/.003
        cost+=5*np.maximum(np.rad2deg(a[:,12])-1.1,0).mean()+.002*np.mean((x/self.bound)**2)
        def longest(mask):
            change=np.diff(np.r_[False,mask,False].astype(int));return float(np.max(np.flatnonzero(change==-1)-np.flatnonzero(change==1),initial=0)*self.dt*5)
        r=dict(cost=float(cost),unstable=False,mean_position_cm=float(a[hold,7].mean()*100),mean_orientation_deg=float(np.rad2deg(a[hold,8]).mean()),left_only_retention_fraction=float(left_alone[hold].mean()),qualified_left_only_fraction=float(qualified[hold].mean()),longest_qualified_left_only_s=longest(qualified),max_joint_excursion_deg=float(np.rad2deg(a[:,12]).max()),max_hand_pen_m=float(a[:,11].max()),timing=timing)
        if return_trace:r['trace']=a
        return r

WORK=None
def initialize(spec):
    global WORK;WORK=HandoverExperiment(spec)
def evaluate(x):return WORK.evaluate(x)

def optimize(spec,output,samples=128,iterations=24,workers=24,seed=361):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    (out/'search_physics').mkdir();_,scene=_freeze(spec['scene'],out/'search_physics');spec=dict(spec,scene=str(scene))
    exp=HandoverExperiment(spec);best=np.zeros(exp.dim);mean=best.copy();std=exp.scale.copy();rng=np.random.default_rng(seed);report=exp.evaluate(best);history=[]
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,dimension=exp.dim,scale=std,bound=exp.bound));dump(out/'initial.json',report);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        for it in range(iterations):
            noise=rng.normal(size=(samples-2,exp.dim))*std
            for j in range(len(noise)):
                if j%4==0:noise[j,28:56]=0
                elif j%4==1:noise[j,:28]=0
            proposals=np.vstack([best,mean,mean+noise]);proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(evaluate,proposals,chunksize=1));order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<report['cost']:best=proposals[order[0]].copy();report=reports[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.15*exp.scale)
            row=dict(iteration=it,best=report,elapsed_s=time.monotonic()-start);history.append(row);dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{it:03d}.json',dict(proposals=proposals,reports=reports));print(json.dumps(row),flush=True)
            if report['mean_position_cm']<2. and report['mean_orientation_deg']<20 and report['left_only_retention_fraction']>.9 and report['qualified_left_only_fraction']>.7:break
    Q,timing=exp.commands(best);assert np.array_equal(Q[:exp.start_frame+1],exp.Q[:exp.start_frame+1])
    for adr in exp.objects.values():assert np.array_equal(Q[:,adr:adr+7],exp.original[:,adr:adr+7])
    path=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Physical bilateral handover optimization; untouched pickup prefix, objects and clock',floating_runtime=True)
    trace=exp.evaluate(best,True).pop('trace');np.savez_compressed(out/'search_trace.npz',trace=trace,columns=np.array(['time','thumb_gap','other_gap','left_both','left_qualified','right_force','nonrobot_support_force','position_error','rotation_error','table_force','self_pen','hand_pen','joint_violation']))
    execution=execute_reference(spec['scene'],path,out/'execution');metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(Path(spec['manifest']).parent),spec=spec,best=report,parameters=best,history=history,reference=str(path),execution=execution,metrics=metrics,learned_policy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for key,value in [('samples',128),('iterations',24),('workers',24),('seed',361)]:p.add_argument('--'+key,type=int,default=value)
    a=vars(p.parse_args());a['spec']=json.loads(a['spec'].read_text());optimize(**a)
