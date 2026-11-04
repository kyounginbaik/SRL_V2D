"""Tune only unused-finger learned-head amplitude/timing in physical rollouts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np,mujoco
from component_policy import ComponentPolicy as PosturePolicy
from learned_feedback_policy import sha
from refine_grasp_cem import Experiment,angular_errors,B,dump
from evaluate_policy_component import evaluate as execute_policy

BASE=B/'refinement_20261018/command_precision/episode_000003/decimals_6/node01/policy.npz'
SCENE=B/'refinement_20261018/episode_000003/unused_fingers_v2/reference_1.00_start3.2/execution/scene_physics.xml'
INP=B/'episodes/episode_000003'

class Search:
 def __init__(self):
  manifest=json.loads((INP/'input_manifest.json').read_text())
  self.metric=Experiment(dict(scene=str(SCENE),reference=str(INP/'tracking_reference.npz'),original_reference=str(INP/'tracking_reference.npz'),source=manifest['source'],geometry=str(INP/'metric_geometry.json')))
  self.m=self.metric.model;self.policy=PosturePolicy(BASE);self.dt=float(self.m.opt.timestep);self.steps=round(1/self.metric.fps/self.dt)
  self.ids=self.policy.data['posture_actuator_ids'];self.groups=np.array([next(i for i,f in enumerate(['middle','ring','pinky']) if f'_{f}_' in self.m.actuator(int(a)).name) for a in self.ids])
  self.fj=np.array([j for j in range(self.m.njnt) if (self.m.joint(j).name or '').startswith('robot_') and any(f'_{f}_' in self.m.joint(j).name for f in ['thumb','index','middle','ring','pinky'])]);self.fq=self.m.jnt_qposadr[self.fj];self.frange=self.m.jnt_range[self.fj]
  self.right={b for b in range(self.m.nbody) if (self.m.body(b).name or '').startswith('robot_right_')}
 def configure(self,x):
  self.policy.data['posture_actuator_scale']=np.asarray(x[:3])[self.groups];self.policy.data['posture_ramp_s']=float(x[3]);self.policy.data['posture_joint_bias_rad']=np.array([x[4] if self.m.actuator(int(a)).name.endswith('_pinky_CMC_position') else x[5] if self.m.actuator(int(a)).name.endswith('_pinky_MCP_AA_position') else 0. for a in self.ids])
  self.policy.data['grasp_wrist_actuator_ids']=np.array([self.m.actuator('robot_right_right_'+n+'_position').id for n in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']]);self.policy.data['grasp_translation_local_m']=np.asarray(x[6:9]);self.policy.data['grasp_rotation_local_rad']=np.asarray(x[9:12]);self.policy.data['grasp_flex_actuator_ids']=np.array([self.m.actuator('robot_right_right_'+n+'_position').id for n in ['thumb_MCP_FE','thumb_IP','index_MCP_FE','index_PIP']]);self.policy.data['grasp_flex_bias_rad']=np.asarray(x[12:16])
 def evaluate(self,x):
  self.configure(x);d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,d,0);mujoco.mj_forward(self.m,d);Q=np.empty_like(self.metric.base);Q[0]=d.qpos;selfpen=0.;limit_violation=0.;unstable=False
  for k in range(1,len(Q)):
   for _ in range(self.steps):
    d.ctrl[:]=self.policy(d.time+self.dt,d.qpos,d.qvel);mujoco.mj_step(self.m,d)
    limit_violation=max(limit_violation,float(np.max(np.maximum(self.frange[:,0]-d.qpos[self.fq],d.qpos[self.fq]-self.frange[:,1]))))
   Q[k]=d.qpos
   if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:unstable=True;break
   for c in d.contact:
    if self.m.geom_bodyid[c.geom1] in self.right and self.m.geom_bodyid[c.geom2] in self.right:selfpen=max(selfpen,-float(c.dist))
  if unstable:return dict(cost=1e6,unstable=True)
  p,q=self.metric.object_poses(Q);ep=np.linalg.norm(p-self.metric.tp,axis=-1);er=angular_errors(q[:,None],self.metric.tq[:,None],self.metric.axes,self.metric.kinds)[:,0]
  pos=ep.mean(0);ang=er.mean(0);ratio=np.maximum(pos/.03,ang/np.deg2rad(30));success=bool(np.all(ratio<1.))
  limit_deg=float(np.degrees(max(0.,limit_violation)))
  cost=float(ratio.max()+.25*ratio.mean()+50*max(0.,selfpen-.002)/.01+.03*np.mean((np.asarray(x[:3])-1.)**2)+100*max(0.,limit_deg-1.1))
  return dict(cost=cost,mean_position_cm=(pos*100).tolist(),mean_orientation_deg=np.degrees(ang).tolist(),mp_sr=float(success),max_right_self_pen_m=selfpen,max_right_finger_limit_violation_deg=limit_deg,posture_quality_pass=limit_deg<=1.1 and selfpen<.002,unstable=False)

WORK=None
def init():
 global WORK;WORK=Search()
def evaluate(x):return WORK.evaluate(x)

def optimize(output,samples=64,iterations=10,workers=16,seed=46):
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise FileExistsError(out)
 search=Search();rng=np.random.default_rng(seed);best=np.asarray(json.loads((B/'refinement_20261018/learned_episode3_posture_grasp/result.json').read_text())['parameters']);mean=best.copy();std=np.r_[[.08,.08,.08,.1,.03,.05],[.0015]*3,[.015]*3,[.015]*4];lo=np.r_[[.25,.25,.25,.6,0.,0.],[-.008]*3,[-.08]*3,[-.06]*4];hi=np.r_[[1.,1.,1.,1.4,.25,.35],[.008]*3,[.08]*3,[.06]*4];r=search.evaluate(best);history=[];start=time.monotonic()
 original=json.loads((BASE.parent/'evaluation/result.json').read_text());expected=original['metrics']['declared_frame_reference_proxy']['objects']
 assert np.allclose(r['mean_position_cm'],[expected[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0),r
 dump(out/'initial.json',r);dump(out/'settings.json',dict(source_policy=str(BASE),scene=str(SCENE),dimensions=['middle_amplitude','ring_amplitude','pinky_amplitude','curl_duration_s','pinky_cmc_inward_bias_rad','pinky_abduction_relief_rad']+[f'local_wrist_translation_{i}' for i in range(3)]+[f'local_wrist_rotation_{i}' for i in range(3)]+['thumb_MCP_FE_bias','thumb_IP_bias','index_MCP_FE_bias','index_PIP_bias'],bounds=[lo,hi],samples=samples,iterations=iterations,workers=workers,seed=seed,scope='Fixed6decimal command precision; moderate unused-digit posture plus bounded wrist and thumb/index countercorrection after3.2s; pre3.2s actor unchanged from the precision-corrected baseline and all physical object targets/clock/state/limits unchanged. Soft-limit tolerance1.1deg includes the fixed passing base actor prefix peak1.017deg; large unused-finger violations are rejected.'))
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
  for it in range(iterations):
   xs=np.vstack([best,mean,np.clip(mean+rng.normal(size=(samples-2,16))*std,lo,hi)]);rs=list(pool.map(evaluate,xs,chunksize=1));order=np.argsort([a['cost'] for a in rs]);elite=xs[order[:max(6,samples//8)]]
   if rs[order[0]]['cost']<r['cost']:best=xs[order[0]].copy();r=rs[order[0]]
   mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),np.r_[[.025,.025,.025,.05,.015,.025],[.0005]*3,[.005]*3,[.004]*4]);row=dict(iteration=it,best=r,parameters=best,elapsed_s=time.monotonic()-start);history.append(row);dump(out/'progress.json',dict(best_parameters=best,history=history));dump(out/f'iteration_{it:03d}.json',dict(proposals=xs,reports=rs));print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
   if r['mp_sr']==1. and r['posture_quality_pass']:break
 search.configure(best);checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**search.policy.data)
 execute_policy(3,str(SCENE),str(checkpoint),str(out/'evaluation'),str(INP));final=json.loads((out/'evaluation/result.json').read_text())
 actual=final['metrics']['declared_frame_reference_proxy']['objects'];assert np.allclose(r['mean_position_cm'],[actual[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0)
 dump(out/'training_report.json',dict(learned_policy=True,method='Physical fine-tuning of learned posture/grasp head with fixed6decimal actuator commands',source_checkpoint=str(BASE),source_checkpoint_sha256=sha(BASE),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),parameters=best,all_finger_joint_limits_checked=True,source_targets_unchanged=True,physics_and_initial_state_unchanged=True))
 dump(out/'result.json',dict(best=r,parameters=best,history=history,checkpoint=str(checkpoint),evaluation=str(out/'evaluation/result.json'),metrics=final['metrics'],physics_verified=final['execution']['physics_verified'],policy_runtime='component_policy.ComponentPolicy'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=10);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=46);optimize(**vars(p.parse_args()))
