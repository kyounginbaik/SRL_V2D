"""Physical suffix search for episode6's learned spoon setdown, unchanged scene."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
import mujoco,numpy as np
from multi_release_feedback_policy import MultiReleasePolicy
from search_supported_release_policy import Experiment as Parent
from evaluate_mp_sr import load_reference,apply_world_transform,rotation_matrices,validate_geometry
from learned_feedback_policy import sha
from physics_cup_pickup_search import dump,H
from evaluate_policy_component import evaluate as final_evaluate
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
B=Path(__file__).resolve().parents[1];SOURCE=B/'refinement_20261018/expanded_batch/bimanual_release/episode_000006/case_0'
class Experiment(Parent):
 def __init__(self):
  self.checkpoint=SOURCE/'policy.npz';self.policy=MultiReleasePolicy(self.checkpoint);self.weights={k:v.copy() for k,v in self.policy.data.items()};self.w=self.policy.data;self.p=str(self.w['multi_release_prefixes'][-1])
  report=json.loads((SOURCE/'evaluation/result.json').read_text());self.scene=Path(report['execution']['scene']);self.m=mujoco.MjModel.from_xml_path(str(self.scene));self.prefix=mujoco.MjData(self.m)
  with np.load(report['execution']['trajectory']) as z:controls=z['substep_ctrl'].copy();initial=z['initial_state'];state_spec=int(z['state_spec']);self.old=z['substep_qpos'].copy()
  self.start=float(self.w[self.p+'start_s']);self.start_step=round(self.start/.002);mujoco.mj_setState(self.m,self.prefix,initial,state_spec)
  for ctrl in controls[:self.start_step]:self.prefix.ctrl[:]=ctrl;mujoco.mj_step(self.m,self.prefix)
  assert np.array_equal(self.prefix.qpos,self.old[self.start_step-1])
  self.qadr=self.policy.qadr;self.jids=self.m.actuator_trnid[:,0];self.limits=self.m.jnt_range[self.jids];self.fingers=np.array([not any(self.m.joint(int(j)).name.endswith(n) for n in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']) for j in self.jids]);assert self.fingers.sum()==44
  self.ob=self.m.body('wooden_spoon').id;self.table=self.m.body('table').id;self.rig=H.HandRig.build(self.m,'robot_right_','right');self.fa=np.array(list(self.rig.finger_act.values()));self.wa=np.array([self.rig.wrist_act[n] for n in ['x','y','z','rx','ry','rz']]);self.robot={i for i in range(self.m.nbody) if self.m.body(i).name.startswith('robot_')}
  ep=B/'episodes/episode_000006';manifest=json.loads((ep/'input_manifest.json').read_text());reference=apply_world_transform(load_reference(manifest['source']),np.asarray(manifest['setup_from_source']));geo=json.loads((ep/'metric_geometry.json').read_text());anchors,_,_=validate_geometry(reference['object_names'],geo);oi=reference['object_names'].index('wooden_spoon');self.anchor=anchors[oi];self.target_rotations=rotation_matrices(reference['object_quaternions_wxyz'])[:,0,oi];self.targets=reference['object_positions'][:,0,oi]+np.einsum('tij,j->ti',self.target_rotations,self.anchor)
  self.dt=.002;self.final_steps=round(self.policy.duration/self.dt)-self.start_step;self.object_dof=int(self.m.joint('wooden_spoon_free').dofadr[0]);self.start_height=float(self.prefix.xpos[self.ob,2]);self.base=self.policy.base
  p=self.p;w=self.w;place=w[p+'place_delta_m'];self.x0=np.r_[place,np.zeros(22),w[p+'open_delta_rad'],w[p+'retreat_m'],[float(w[p+'place_duration_s']),float(w[p+'open_duration_s'])],np.zeros(3)]
  self.scale=np.r_[[.012]*3,[.035]*22,[.07]*22,[.02]*3,[.1,.05],[.10]*3]
  self.lo=np.r_[place-.06,[-.25]*22,[-2.]*22,[-.15,-.15,.015],[.2,.12],[-.65]*3]
  self.hi=np.r_[place+.06,[.25]*22,[2.]*22,[.15,.15,.18],[1.1,.6],[.65]*3];self.x0=np.clip(self.x0,self.lo,self.hi)
 def set(self,x):
  p=self.p;w=self.w;w[p+'place_delta_m']=x[:3];w[p+'closed_bias_rad']=x[3:25];w[p+'open_delta_rad']=x[25:47];w[p+'retreat_m']=x[47:50];w[p+'place_duration_s']=np.array(x[50]);w[p+'open_duration_s']=np.array(x[51]);w[p+'retreat_duration_s']=np.array(.8);w[p+'place_rotation_vector']=x[52:55]
_EXP=None
def init():
 global _EXP;_EXP=Experiment()
def score(x):return _EXP.evaluate(x)
def run(output,samples=64,iterations=12,workers=16,seed=64):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 if (out/'settings.json').exists():raise FileExistsError(out)
 exp=Experiment();rng=np.random.default_rng(seed);mean=exp.x0.copy();std=exp.scale.copy();bestx=mean.copy();best=exp.evaluate(mean);history=[];started=time.monotonic();dump(out/'initial.json',best)
 dump(out/'settings.json',dict(source=str(SOURCE),samples=samples,iterations=iterations,workers=workers,seed=seed,protected_prefix_s=exp.start,scope='Learned finite robot release parameters, actual contact dynamics, unchanged source objects and timing'))
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
  for iteration in range(iterations):
   X=np.clip(mean+rng.normal(size=(samples,len(mean)))*std,exp.lo,exp.hi);X[0]=bestx;X[1]=mean;reports=list(pool.map(score,X,chunksize=1));order=np.argsort([r['cost'] for r in reports]);elite=X[order[:max(6,samples//8)]]
   if reports[order[0]]['cost']<best['cost']:best=reports[order[0]];bestx=X[order[0]].copy()
   mean=.25*mean+.75*elite.mean(0);std=np.maximum(.45*std+.55*elite.std(0),exp.scale*.12);row=dict(iteration=iteration,best=best,elapsed_s=time.monotonic()-started);history.append(row);dump(out/'progress.json',dict(best=best,parameters=bestx,history=history));print(json.dumps(row),flush=True)
 exp.set(bestx);checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**exp.w)
 dump(out/'training_report.json',dict(method='Physics-fitted finite placement, grip preload, opening and withdrawal parameters on learned feedback actor',source_checkpoint=str(exp.checkpoint),source_checkpoint_sha256=sha(exp.checkpoint),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),parameters=bestx,protected_prefix_s=exp.start,runtime_inputs=['phase','measured robot positions','measured robot velocities'],no_object_trajectory_or_command_table=True,original_learned_weights_unchanged=True,scene_limits_and_original_targets_unchanged=True))
 result=final_evaluate(6,exp.scene,checkpoint,out/'evaluation',B/'episodes/episode_000006');ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);dump(ad/'actual_joint_limits.json',joint_audit(result['execution']['scene'],result['execution']['trajectory']));support_audit(out/'evaluation')
 with np.load(result['execution']['trajectory']) as z:err=float(np.max(abs(z['substep_qpos'][:exp.start_step]-exp.old[:exp.start_step])))
 assert err<=1e-10;dump(out/'result.json',dict(best=best,parameters=bestx,evaluation=str(out/'evaluation/result.json'),prefix_max_qpos_error=err))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=64);run(**vars(p.parse_args()))
