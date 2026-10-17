"""Tune only unused-finger learned-head amplitude/timing in physical rollouts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np,mujoco
from posture_feedback_policy import PosturePolicy
from refine_grasp_cem import Experiment,angular_errors,B,dump
from evaluate_posture_feedback import main as execute_policy

BASE=B/'refinement_20261018/learned_episode3_posture_head/training/policy.npz'
SCENE=B/'refinement_20261018/episode_000003/unused_fingers_v2/reference_1.00_start3.2/execution/scene_physics.xml'
INP=B/'episodes/episode_000003'

class Search:
 def __init__(self):
  manifest=json.loads((INP/'input_manifest.json').read_text())
  self.metric=Experiment(dict(scene=str(SCENE),reference=str(INP/'tracking_reference.npz'),original_reference=str(INP/'tracking_reference.npz'),source=manifest['source'],geometry=str(INP/'metric_geometry.json')))
  self.m=self.metric.model;self.policy=PosturePolicy(BASE);self.dt=float(self.m.opt.timestep);self.steps=round(1/self.metric.fps/self.dt)
  self.ids=self.policy.data['posture_actuator_ids'];self.groups=np.array([next(i for i,f in enumerate(['middle','ring','pinky']) if f'_{f}_' in self.m.actuator(int(a)).name) for a in self.ids])
  self.right={b for b in range(self.m.nbody) if (self.m.body(b).name or '').startswith('robot_right_')}
 def configure(self,x):
  self.policy.data['posture_actuator_scale']=np.asarray(x[:3])[self.groups];self.policy.data['posture_ramp_s']=float(x[3])
 def evaluate(self,x):
  self.configure(x);d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,d,0);mujoco.mj_forward(self.m,d);Q=np.empty_like(self.metric.base);Q[0]=d.qpos;selfpen=0.;unstable=False
  for k in range(1,len(Q)):
   for _ in range(self.steps):
    d.ctrl[:]=self.policy(d.time+self.dt,d.qpos,d.qvel);mujoco.mj_step(self.m,d)
   Q[k]=d.qpos
   if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:unstable=True;break
   for c in d.contact:
    if self.m.geom_bodyid[c.geom1] in self.right and self.m.geom_bodyid[c.geom2] in self.right:selfpen=max(selfpen,-float(c.dist))
  if unstable:return dict(cost=1e6,unstable=True)
  p,q=self.metric.object_poses(Q);ep=np.linalg.norm(p-self.metric.tp,axis=-1);er=angular_errors(q[:,None],self.metric.tq[:,None],self.metric.axes,self.metric.kinds)[:,0]
  pos=ep.mean(0);ang=er.mean(0);ratio=np.maximum(pos/.03,ang/np.deg2rad(30));success=bool(np.all(ratio<1.))
  cost=float(ratio.max()+.25*ratio.mean()+50*max(0.,selfpen-.002)/.01+.03*np.mean((np.asarray(x[:3])-1.)**2))
  return dict(cost=cost,mean_position_cm=(pos*100).tolist(),mean_orientation_deg=np.degrees(ang).tolist(),mp_sr=float(success),max_right_self_pen_m=selfpen,unstable=False)

WORK=None
def init():
 global WORK;WORK=Search()
def evaluate(x):return WORK.evaluate(x)

def optimize(output,samples=64,iterations=10,workers=16,seed=46):
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise FileExistsError(out)
 search=Search();rng=np.random.default_rng(seed);best=np.array([1.,1.,1.,.6]);mean=best.copy();std=np.array([.18,.18,.18,.4]);lo=np.array([.6,.6,.6,.6]);hi=np.array([1.2,1.2,1.2,1.8]);r=search.evaluate(best);history=[];start=time.monotonic()
 original=json.loads((BASE.parent.parent/'evaluation/result.json').read_text());expected=original['metrics']['declared_frame_reference_proxy']['objects']
 assert np.allclose(r['mean_position_cm'],[expected[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0),r
 dump(out/'initial.json',r);dump(out/'settings.json',dict(source_policy=str(BASE),scene=str(SCENE),dimensions=['middle_amplitude','ring_amplitude','pinky_amplitude','curl_duration_s'],bounds=[lo,hi],samples=samples,iterations=iterations,workers=workers,seed=seed,scope='Only gated unused-digit posture parameters; physical object targets/clock/state/active actor outputs unchanged'))
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
  for it in range(iterations):
   xs=np.vstack([best,mean,np.clip(mean+rng.normal(size=(samples-2,4))*std,lo,hi)]);rs=list(pool.map(evaluate,xs,chunksize=1));order=np.argsort([a['cost'] for a in rs]);elite=xs[order[:max(6,samples//8)]]
   if rs[order[0]]['cost']<r['cost']:best=xs[order[0]].copy();r=rs[order[0]]
   mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),[.025,.025,.025,.05]);row=dict(iteration=it,best=r,parameters=best,elapsed_s=time.monotonic()-start);history.append(row);dump(out/'progress.json',dict(best_parameters=best,history=history));dump(out/f'iteration_{it:03d}.json',dict(proposals=xs,reports=rs));print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
   if r['mp_sr']==1. and r['max_right_self_pen_m']<.002:break
 search.configure(best);checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**search.policy.data)
 execute_policy(3,str(SCENE),str(checkpoint),str(out/'evaluation'));final=json.loads((out/'evaluation/result.json').read_text())
 actual=final['metrics']['declared_frame_reference_proxy']['objects'];assert np.allclose(r['mean_position_cm'],[actual[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0)
 dump(out/'result.json',dict(best=r,parameters=best,history=history,checkpoint=str(checkpoint),evaluation=str(out/'evaluation/result.json'),metrics=final['metrics'],physics_verified=final['execution']['physics_verified'],policy_runtime='posture_feedback_policy.PosturePolicy'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=10);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=46);optimize(**vars(p.parse_args()))
