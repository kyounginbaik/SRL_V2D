"""Optimize a short learned-control release suffix in actual contact dynamics.

The immutable0--14.4s physics prefix is cached exactly. Search parameters change
only finite robot placement/opening/withdrawal commands in ReleasePolicy.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,os,sys,time,multiprocessing as mp
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from release_feedback_policy import ReleasePolicy
from component_policy import ComponentPolicy
from learned_feedback_policy import sha
from evaluate_mp_sr import load_reference,apply_world_transform,rotation_matrices,validate_geometry
from physics_cup_pickup_search import dump,H
import evaluate_policy_component as evaluator
from audit_full_joint_limits import main as joint_audit
from audit_object_support import audit

B=Path(__file__).resolve().parents[1];ROOT=B/'refinement_20261018';START=14.4

class Experiment:
    def __init__(self, checkpoint):
        self.checkpoint=Path(checkpoint);self.policy=ReleasePolicy(checkpoint);self.weights={k:v.copy() for k,v in self.policy.data.items()};self.w=self.policy.data
        original=ROOT/'command_precision/episode_000003/decimals_6/node01/evaluation';report=json.loads((original/'result.json').read_text());self.scene=Path(report['execution']['scene']);self.m=mujoco.MjModel.from_xml_path(str(self.scene));self.prefix=mujoco.MjData(self.m)
        with np.load(report['execution']['trajectory']) as z:controls=z['substep_ctrl'].copy();initial=z['initial_state'];state_spec=int(z['state_spec']);self.old=z['substep_qpos'].copy()
        mujoco.mj_setState(self.m,self.prefix,initial,state_spec);self.start_step=round(START/.002)
        for ctrl in controls[:self.start_step]:self.prefix.ctrl[:]=ctrl;mujoco.mj_step(self.m,self.prefix)
        assert np.array_equal(self.prefix.qpos,self.old[self.start_step-1])
        self.qadr=self.policy.qadr;self.jids=self.m.actuator_trnid[:,0];self.limits=self.m.jnt_range[self.jids];self.fingers=np.array([not any(self.m.joint(int(j)).name.endswith(n) for n in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']) for j in self.jids]);self.ob=self.m.body('wooden_spoon').id;self.table=self.m.body('table').id
        assert self.fingers.sum()==44, 'Every actuated finger joint must be included in the physical limit objective'
        self.rig=H.HandRig.build(self.m,'robot_right_','right');self.fa=np.array(list(self.rig.finger_act.values()));self.wa=np.array([self.rig.wrist_act[n] for n in ['x','y','z','rx','ry','rz']]);self.robot={i for i in range(self.m.nbody) if self.m.body(i).name.startswith('robot_')}
        ep=B/'episodes/episode_000003';manifest=json.loads((ep/'input_manifest.json').read_text());reference=apply_world_transform(load_reference(manifest['source']),np.asarray(manifest['setup_from_source']));geo=json.loads((ep/'metric_geometry.json').read_text());anchors,_,_=validate_geometry(reference['object_names'],geo);oi=reference['object_names'].index('wooden_spoon');self.anchor=anchors[oi];self.target_rotations=rotation_matrices(reference['object_quaternions_wxyz'])[:,0,oi];self.targets=reference['object_positions'][:,0,oi]+np.einsum('tij,j->ti',self.target_rotations,self.anchor)
        self.dt=.002;self.final_steps=round(self.policy.duration/self.dt)-self.start_step
        self.object_dof=int(self.m.joint('wooden_spoon_free').dofadr[0]);self.start_height=float(self.prefix.xpos[self.ob,2])
        self.base=self.policy.base;self.nominal=self.base(START,self.prefix.qpos,self.prefix.qvel)
        self.natural=self.base(self.policy.duration,self.prefix.qpos,self.prefix.qvel)
        #53 dimensions: placement3,closed22,opened22,withdrawal3,placement/open durations2.
        desired_world_delta=self.target_rotations[-1]@self.prefix.xmat[self.ob].reshape(3,3).T
        wristR=Rotation.from_euler('XYZ',self.nominal[self.wa[3:]]).as_matrix()
        rotation_seed=Rotation.from_matrix(wristR.T@desired_world_delta@wristR).as_rotvec()
        self.x0=np.r_[[-.014,.014,-.030],np.zeros(22),self.natural[self.fa]-self.nominal[self.fa],[.06,-.06,.09],[.2,.3],rotation_seed]
        self.scale=np.r_[[.01]*3,[.05]*22,[.10]*22,[.015]*3,[.06,.06],[.08]*3]
        self.lo=np.r_[[-.06,-.02,-.065],[-.35]*22,[-2.]*22,[-.05,-.15,.025],[.08,.1],[-.65]*3]
        self.hi=np.r_[[.02,.07,.015],[.35]*22,[2.]*22,[.15,.05,.18],[.45,.5],[.65]*3]
        self.x0=np.clip(self.x0,self.lo,self.hi)

    def set(self,x):
        x=np.asarray(x);w=self.w;w['release_actuator_ids']=np.r_[self.wa,self.fa];w['release_finger_actuator_ids']=self.fa;w['release_translation_actuator_ids']=self.wa[:3]
        w['release_place_delta_m']=x[:3];w['release_closed_bias_rad']=x[3:25];w['release_open_delta_rad']=x[25:47];w['release_retreat_m']=x[47:50];w['release_place_duration_s']=np.array(x[50]);w['release_open_duration_s']=np.array(x[51]);w['release_retreat_duration_s']=np.array(max(.15,self.policy.duration-START-x[50]-x[51]))
        w['release_rotation_actuator_ids']=self.wa[3:];w['release_place_rotation_vector']=x[52:55]

    def evaluate(self,x):
        self.set(x);d=mujoco.MjData(self.m);mujoco.mj_copyData(d,self.m,self.prefix);force=np.zeros(6);rows=[];violation=0.;hand_table=0.;unstable=False;max_rise=0.;max_up=0.;max_spin=0.;ever_supported=False
        for step in range(self.final_steps):
            d.ctrl[:]=self.policy(d.time+self.dt,d.qpos,d.qvel);mujoco.mj_step(self.m,d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:unstable=True;break
            max_rise=max(max_rise,float(d.xpos[self.ob,2]-self.start_height));max_up=max(max_up,float(d.qvel[self.object_dof+2]));max_spin=max(max_spin,float(np.linalg.norm(d.qvel[self.object_dof+3:self.object_dof+6])))
            j=d.qpos[self.qadr];over=np.maximum(self.limits[:,0]-j,j-self.limits[:,1]);violation=max(violation,float(np.max(over[self.fingers],initial=0.)))
            if step%5!=4:continue
            support=0.;touch=0.;selfpen=0.
            for i,c in enumerate(d.contact):
                b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2]);r1=b1 in self.robot;r2=b2 in self.robot
                if r1 and r2:selfpen=max(selfpen,-float(c.dist))
                if self.ob not in [b1,b2] and not ((r1 and b2==self.table)or(r2 and b1==self.table)):continue
                mujoco.mj_contactForce(self.m,d,i,force);fn=max(0.,float(force[0]))
                if self.ob in [b1,b2]:
                    other=b2 if b1==self.ob else b1
                    if other==self.table:support+=fn
                    if other in self.robot:touch+=fn
                else:hand_table=max(hand_table,fn)
            p=d.xpos[self.ob]+d.xmat[self.ob].reshape(3,3)@self.anchor;target=self.targets[min(len(self.targets)-1,round(d.time*20))];error=np.linalg.norm(p-target)
            angle=np.degrees(Rotation.from_matrix(self.target_rotations[min(len(self.targets)-1,round(d.time*20))].T@d.xmat[self.ob].reshape(3,3)).magnitude())
            ever_supported|=support>.1
            rows.append([d.time,error,support,touch,selfpen,angle,float(touch<.1 and not ever_supported)])
        if unstable:return dict(cost=1e5,unstable=True)
        a=np.asarray(rows);end=a[:,0]>=self.policy.duration-.25;support_fraction=float(np.mean(a[end,2]>.1));free_fraction=float(np.mean(a[end,3]<.1));deg=np.degrees(violation)
        cost=150*np.mean(a[:,1])+150*np.mean(a[end,1])+.1*np.mean(a[:,5])+.15*np.mean(a[end,5])+25*(1-support_fraction)+20*(1-free_fraction)+.15*np.mean(np.minimum(a[end,3],100))+25*max(0.,deg-1.05)**2+2*max(0.,hand_table-2)+1000*max(0.,np.max(a[:,4])-.0015)+1000*max(0.,max_rise-.005)+30*max(0.,max_up-.12)+2*max(0.,max_spin-3.)+20*np.mean(a[:,6])
        return dict(cost=float(cost),mean_suffix_position_cm=float(100*np.mean(a[:,1])),mean_suffix_orientation_deg=float(np.mean(a[:,5])),final_position_cm=float(100*a[-1,1]),final_supported_fraction=support_fraction,final_hand_free_fraction=free_fraction,max_finger_violation_deg=float(deg),max_hand_table_force_n=float(hand_table),max_object_rise_cm=100*max_rise,max_upward_velocity_m_s=max_up,max_object_spin_rad_s=max_spin,unsupported_initial_release_fraction=float(np.mean(a[:,6])),unstable=False)

_EXP=None
def init(checkpoint):
    global _EXP;_EXP=Experiment(checkpoint)
def evaluate(x):return _EXP.evaluate(x)

def run(output,samples,iterations,workers,seed,warm_start):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    checkpoint=ROOT/'episode_000003/quant6_place_release/policy.npz';exp=Experiment(checkpoint);rng=np.random.default_rng(seed);mean=exp.x0.copy();std=exp.scale.copy()
    if warm_start:
        previous=np.asarray(json.loads(Path(warm_start).read_text())['best_parameters']);mean[:len(previous)]=previous
        if len(previous)==52:mean[52:]=0. # The previous incumbent must remain a valid baseline.
        mean=np.clip(mean,exp.lo,exp.hi)
    bestx=mean.copy();best=exp.evaluate(mean);history=[];started=time.monotonic()
    dump(out/'settings.json',dict(method='Contact-dynamics optimization of finite learned release-actor parameters',seed=seed,samples=samples,iterations=iterations,workers=workers,protected_prefix_s=START,source_checkpoint=str(checkpoint),initial_parameters=mean,parameter_dim=len(mean)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(checkpoint),))as pool:
        for iteration in range(iterations):
            X=np.clip(mean+rng.normal(size=(samples,len(mean)))*std,exp.lo,exp.hi);X[0]=bestx;X[1]=mean;results=list(pool.map(evaluate,X,chunksize=1));order=np.argsort([r['cost']for r in results]);elite=X[order[:max(6,samples//8)]]
            if results[order[0]]['cost']<best['cost']:best=results[order[0]];bestx=X[order[0]].copy()
            mean=.25*mean+.75*elite.mean(0);std=np.maximum(.45*std+.55*elite.std(0),exp.scale*.12)
            row=dict(iteration=iteration,best=best,elapsed_s=time.monotonic()-started);history.append(row);dump(out/'progress.json',dict(best=best,best_parameters=bestx,history=history));print(json.dumps(row),flush=True)
            if best['final_supported_fraction']>.95 and best['final_hand_free_fraction']>.95 and best['max_finger_violation_deg']<1.1 and best['mean_suffix_position_cm']<4 and best['max_object_rise_cm']<.8 and best['max_upward_velocity_m_s']<.15 and best['unsupported_initial_release_fraction']<.02:break
    exp.set(bestx);saved=dict(exp.weights);saved.update(exp.w);path=out/'policy.npz';np.savez_compressed(path,**saved)
    dump(out/'training_report.json',dict(method='Physical suffix optimization of learned phase/proprioception release actor',source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),checkpoint=str(path),checkpoint_sha256=sha(path),parameters=bestx,settings=str(out/'settings.json'),progress=str(out/'progress.json'),best_suffix_metrics=best,protected_prefix_s=START,original_scene_and_reference_targets_unchanged=True,runtime_inputs=['phase','actual robot qpos','actual robot qvel']))
    evaluator.LearnedPolicy=ReleasePolicy;result=evaluator.evaluate(3,exp.scene,path,out/'evaluation',B/'episodes/episode_000003');joint_audit(out/'evaluation');audit(out/'evaluation')
    with np.load(result['execution']['trajectory'])as z:err=float(abs(z['substep_qpos'][:exp.start_step]-exp.old[:exp.start_step]).max())
    assert err==0.;dump(out/'prefix_validation.json',dict(prefix_until_s=START,max_qpos_error=err));dump(out/'result.json',dict(best=best,parameters=bestx,evaluation=str(out/'evaluation/result.json'),prefix_max_qpos_error=err))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=16);p.add_argument('--workers',type=int,default=24);p.add_argument('--seed',type=int,default=44);p.add_argument('--warm-start',type=Path);run(**vars(p.parse_args()))
