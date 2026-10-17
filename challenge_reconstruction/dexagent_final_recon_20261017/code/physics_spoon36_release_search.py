"""Physical suffix search for genuinely hands-free, supported spoon release.

The complete successful carry prefix is fixed. Only bounded robot wrist and
finger targets after18.3s change. The physical object is never actuated or
reset, and all original target poses and timestamps remain unchanged.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search import B,H,TR,blend,dump,execute_reference,evaluate_files
from wrist_reference import continuous_euler_xyz
from audit_object_support import audit as audit_support
from audit_full_joint_limits import main as audit_joints

class ReleaseExperiment:
    def __init__(self,spec):
        self.spec=spec;self.m=mujoco.MjModel.from_xml_path(spec['scene']);self.ref=TR.load_reference(spec['reference'],self.m);self.Q=self.ref['qpos'];self.fps=self.ref['fps'];self.times=np.arange(len(self.Q))/self.fps
        self.rig=H.HandRig.build(self.m,'robot_right_','right');self.wc=np.array([self.rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);self.fc=np.array(list(self.rig.finger_qpos.values()));self.cols=np.r_[self.wc,self.fc]
        self.acts=np.array([self.rig.wrist_act[k] for k in ['x','y','z','rx','ry','rz']]+list(self.rig.finger_act.values()));self.rot=Rotation.from_euler('XYZ',self.Q[:,self.wc[3:]])
        self.qadr=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]];self.lo,self.hi=self.m.actuator_ctrlrange.T
        self.ob=self.m.body('wooden_spoon').id;self.oc=int(self.m.joint('wooden_spoon_free').qposadr[0]);self.robot={i for i in range(self.m.nbody) if self.m.body(i).name.startswith('robot_')}
        self.fj=np.array([j for j in range(self.m.njnt) if self.m.joint(j).name.startswith('robot_') and any(f'_{f}_' in self.m.joint(j).name for f in H.FINGERS)]);assert len(self.fj)==44
        self.start=18.3;self.dt=float(self.m.opt.timestep);self.first=round(self.start/self.dt);self.last=round(self.times[-1]/self.dt);self.k=round(self.start*self.fps);self.steps_per_frame=round(1/self.fps/self.dt)
        self.prefix=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,self.prefix,0);mujoco.mj_forward(self.m,self.prefix)
        for step in range(self.first):self.prefix.ctrl[:]=self.control(self.Q,step);mujoco.mj_step(self.m,self.prefix)
        with np.load(Path(spec['scene']).with_name('trajectory.npz')) as recorded:
            if not np.array_equal(self.prefix.qpos,recorded['substep_qpos'][self.first-1]):raise AssertionError('Cached physical prefix differs from exact source replay')
        self.dim=34;self.scale=np.r_[[.012]*3,[.16]*3,[.22]*22,[.025]*3,[.20]*3];self.bound=np.r_[[.10]*3,[1.1]*3,[1.3]*22,[.20]*3,[1.5]*3]
        self.target_R=Rotation.from_quat(self.Q[:,self.oc+3:self.oc+7],scalar_first=True)

    def control(self,Q,step):
        i=step//self.steps_per_frame;a=(step%self.steps_per_frame+1)/self.steps_per_frame
        return np.clip(Q[i,self.qadr]+a*(Q[i+1,self.qadr]-Q[i,self.qadr]),self.lo,self.hi)

    def commands(self,x):
        Q=self.Q.copy();a=blend((self.times-self.start)/.45);b=blend((self.times-18.8)/.8)
        Q[:,self.wc[:3]]+=a[:,None]*x[:3]+b[:,None]*x[28:31]
        if np.any(x[3:6]) or np.any(x[31:34]):
            R=self.rot*Rotation.from_rotvec(a[:,None]*x[3:6]+b[:,None]*x[31:34]);Q[:,self.wc[3:]]=continuous_euler_xyz(R,self.Q[0,self.wc[3:]])
        Q[:,self.fc]+=a[:,None]*x[6:28]
        for act in self.acts:Q[:,self.qadr[act]]=np.clip(Q[:,self.qadr[act]],self.lo[act],self.hi[act])
        Q[:self.k+1]=self.Q[:self.k+1]
        return Q

    def evaluate(self,x,trace=False):
        Q=self.commands(x);d=mujoco.MjData(self.m);mujoco.mj_copyData(d,self.m,self.prefix);force=np.zeros(6);rows=[]
        for step in range(self.first,self.last):
            d.ctrl[:]=self.control(Q,step);mujoco.mj_step(self.m,d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:return dict(cost=1e9,unstable=True)
            if step%5!=4:continue
            frame=min(len(Q)-1,round(d.time*self.fps));hand=0.;support=0.;table=0.;selfpen=0.;handpen=0.
            for i,c in enumerate(d.contact):
                a,b=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2]);ra,rb=a in self.robot,b in self.robot
                if ra and rb:selfpen=max(selfpen,-float(c.dist))
                if ra!=rb:
                    handpen=max(handpen,-float(c.dist));other=b if ra else a
                    if self.m.body(other).name=='table':mujoco.mj_contactForce(self.m,d,i,force);table+=max(0.,float(force[0]))
                other=b if a==self.ob else a if b==self.ob else -1
                if other<0:continue
                mujoco.mj_contactForce(self.m,d,i,force);fn=max(0.,float(force[0]))
                if other in self.robot:hand+=fn
                else:support+=fn
            p=np.linalg.norm(d.qpos[self.oc:self.oc+3]-self.Q[frame,self.oc:self.oc+3]);r=(Rotation.from_quat(d.qpos[self.oc+3:self.oc+7],scalar_first=True).inv()*self.target_R[frame]).magnitude()
            f=d.qpos[self.m.jnt_qposadr[self.fj]];ranges=self.m.jnt_range[self.fj];exc=float(np.maximum(np.maximum(ranges[:,0]-f,f-ranges[:,1]),0.).max())
            rows.append([d.time,p,r,hand,support,table,selfpen,handpen,exc,d.qpos[self.oc+2]])
        a=np.array(rows);end=a[:,0]>=19.8;late=a[:,0]>=18.6
        pos=a[:,1]/.03;angle=a[:,2]/np.deg2rad(30)
        cost=2*np.minimum(pos[late],8).mean()+np.minimum(angle[late],6).mean()+5*np.minimum(pos[end],8).mean()+2*np.minimum(angle[end],6).mean()
        cost+=10*np.minimum(a[end,3]/.5,8).mean()+10*np.mean(a[end,4]<.1)
        cost+=.1*np.minimum(a[:,5]/50,10).mean()+15*np.maximum(a[:,6]-.002,0).mean()/.002+15*np.maximum(a[:,7]-.003,0).mean()/.003
        cost+=10*np.maximum(np.rad2deg(a[:,8])-1.1,0).mean()+.005*np.mean((x/self.bound)**2)
        report=dict(cost=float(cost),unstable=False,late_mean_position_cm=float(a[late,1].mean()*100),late_mean_orientation_deg=float(np.rad2deg(a[late,2]).mean()),final_mean_position_cm=float(a[end,1].mean()*100),final_mean_orientation_deg=float(np.rad2deg(a[end,2]).mean()),final_max_hand_force_n=float(a[end,3].max()),final_min_support_n=float(a[end,4].min()),final_hands_free_fraction=float(np.mean(a[end,3]<.05)),max_finger_excursion_deg=float(np.rad2deg(a[:,8]).max()),max_hand_pen_m=float(a[:,7].max()))
        if trace:report['trace']=a
        return report

WORK=None
def init(spec):
    global WORK;WORK=ReleaseExperiment(spec)
def evaluate(x):return WORK.evaluate(x)

def main(output,samples=128,iterations=25,workers=16,seed=3660):
    source=B/'refinement_20261018/episode_000036/right_only_thumb_release/open_0.016';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    spec=dict(scene=str(source/'execution/scene_physics.xml'),reference=str(source/'reference.npz'));exp=ReleaseExperiment(spec);rng=np.random.default_rng(seed);best=np.zeros(exp.dim);mean=best.copy();std=exp.scale.copy();report=exp.evaluate(best);history=[];begin=time.monotonic()
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,scale=std,bound=exp.bound,objective='Physical hands-free supported release with original targets and exact carry prefix'));dump(out/'initial.json',report)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for it in range(iterations):
            noise=rng.normal(size=(samples-2,exp.dim))*std
            for j in range(len(noise)):
                if j%4==0:noise[j,6:28]=0
                elif j%4==1:noise[j,:6]=0;noise[j,28:]=0
            xs=np.vstack([best,mean,np.clip(mean+noise,-exp.bound,exp.bound)]);rs=list(pool.map(evaluate,xs,chunksize=1));order=np.argsort([r['cost'] for r in rs]);elite=xs[order[:max(6,samples//8)]]
            if rs[order[0]]['cost']<report['cost']:best=xs[order[0]].copy();report=rs[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.12*exp.scale);row=dict(iteration=it,best=report,elapsed_s=time.monotonic()-begin);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{it:03d}.json',dict(proposals=xs,reports=rs));print(json.dumps(row),flush=True)
            if report['final_max_hand_force_n']<.01 and report['final_min_support_n']>.2 and report['final_mean_position_cm']<3 and report['final_mean_orientation_deg']<30:break
    Q=exp.commands(best);assert np.array_equal(Q[:exp.k+1],exp.Q[:exp.k+1]);other=np.setdiff1d(np.arange(exp.m.nq),exp.cols);assert np.array_equal(Q[:,other],exp.Q[:,other])
    reference=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Physical release optimization with exact carry prefix and immutable object targets; right-only alternative strategy',floating_runtime=True)
    full=exp.evaluate(best,True);np.savez_compressed(out/'search_trace.npz',trace=full.pop('trace'),columns=np.array(['time','position_error','rotation_error','hand_force','support_force','table_force','self_pen','hand_pen','joint_excursion','object_origin_z']))
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text());execution=execute_reference(spec['scene'],reference,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',protected_prefix_s=exp.start,parameters=best,best=report,history=history,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    audit_support(out);audit_joints(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    for k,v in [('samples',128),('iterations',25),('workers',16),('seed',3660)]:p.add_argument('--'+k,type=int,default=v)
    main(**vars(p.parse_args()))
