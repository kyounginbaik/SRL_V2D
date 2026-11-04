"""Learn only a smooth, gated unused-digit posture head on a passing BC actor.

All active grasp/wrist/left-hand output functions remain identical to the base
actor. Until the gate opens, all56 output commands are exactly the base actor.
No reference trajectory or future state is available during policy execution.
"""
from pathlib import Path
import argparse,json
import numpy as np,scipy.linalg,mujoco
from learned_feedback_policy import LearnedPolicy,features,sha

SCHEMA='radial_basis_behavior_cloning_with_gated_posture_head'

def smooth(u):
 u=np.clip(u,0.,1.);return u**3*(10.-15.*u+6.*u*u)

class PosturePolicy(LearnedPolicy):
 def __init__(self,checkpoint):
  with np.load(checkpoint,allow_pickle=False) as z:self.data={k:z[k] for k in z.files}
  if str(self.data['algorithm'])!=SCHEMA:raise ValueError('Unexpected posture-head policy format')
  self.duration=float(self.data['duration_s']);self.qadr=self.data['qadr'];self.vadr=self.data['vadr']
 def gate(self,t):
  d=self.data;return smooth((t-float(d['posture_start_s']))/float(d['posture_ramp_s']))*(1-smooth((t-float(d['posture_release_s']))/float(d['posture_release_ramp_s'])))
 def __call__(self,next_time_s,qpos,qvel):
  output=super().__call__(next_time_s,qpos,qvel);g=self.gate(next_time_s)
  if g<=0:return output
  d=self.data;phi=features([next_time_s/self.duration],d['centers'],float(d['width']))[0]
  ids=d['posture_actuator_ids'];output[ids]+=g*(phi@d['posture_weights'])*d.get('posture_actuator_scale',1.)
  return np.clip(output,d['ctrl_lo'],d['ctrl_hi'])

def train(base_checkpoint,scene,trajectory,output,start_s=3.2):
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);checkpoint=out/'policy.npz'
 if checkpoint.exists():raise FileExistsError(checkpoint)
 base=LearnedPolicy(base_checkpoint);m=mujoco.MjModel.from_xml_path(str(scene))
 if str(base.data['scene_sha256'])!=sha(scene):raise ValueError('Base actor and physical scene differ')
 with np.load(trajectory,allow_pickle=False) as z:
  ctrl=z['substep_ctrl'];q=np.vstack([z['initial_qpos'][None],z['substep_qpos'][:-1]]);v=np.vstack([z['initial_qvel'][None],z['substep_qvel'][:-1]]);t=z['substep_time_s']
  if str(z['scene_sha256'])!=sha(scene):raise ValueError('Teacher scene mismatch')
 ids=np.array([a for a in range(m.nu) if (m.actuator(a).name or '').startswith('robot_right_') and any(f'_{f}_' in m.actuator(a).name for f in ['middle','ring','pinky'])]);assert len(ids)==13
 predicted=np.vstack([base(float(ti),qi,vi) for ti,qi,vi in zip(t,q,v)])
 gate=smooth((t-start_s)/.6)*(1-smooth((t-14.45)/.35));Phi=features(t/base.duration,base.data['centers'],float(base.data['width']));X=Phi*gate[:,None]
 target=ctrl[:,ids]-predicted[:,ids];held=np.arange(len(t))%7==3;good=~held
 weights=scipy.linalg.solve(X[good].T@X[good]+1e-9*np.eye(X.shape[1]),X[good].T@target[good],assume_a='pos')
 fit=X@weights;data=dict(base.data);data.update(algorithm=np.array(SCHEMA),posture_actuator_ids=ids,posture_weights=weights,posture_start_s=start_s,posture_ramp_s=.6,posture_release_s=14.45,posture_release_ramp_s=.35)
 np.savez_compressed(checkpoint,**data);policy=PosturePolicy(checkpoint);others=np.setdiff1d(np.arange(m.nu),ids)
 maxprotected=0.;maxprefix=0.
 for ti,qi,vi,pi in zip(t,q,v,predicted):
  action=policy(float(ti),qi,vi);maxprotected=max(maxprotected,float(np.max(np.abs(action[others]-pi[others]))))
  if ti<=start_s:maxprefix=max(maxprefix,float(np.max(np.abs(action-pi))))
 assert maxprotected==0. and maxprefix==0.
 report=dict(algorithm=SCHEMA,base_checkpoint=str(Path(base_checkpoint).resolve()),base_checkpoint_sha256=sha(base_checkpoint),scene=str(Path(scene).resolve()),scene_sha256=sha(scene),demonstration=str(Path(trajectory).resolve()),demonstration_sha256=sha(trajectory),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trained_actuator_names=[m.actuator(int(a)).name for a in ids],protected_action_function_max_error=maxprotected,pre3p2_base_action_function_max_error=maxprefix,heldout_posture_head_max_error_rad=float(np.max(np.abs((fit-target)[held&(gate>.99)]))),posture_head_only=True,learned_policy=True,contains_command_table=False,training_samples=int(good.sum()),heldout_scope='Interleaved teacher time; full physical evaluation still required',trainer_sha256=sha(__file__))
 (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base-checkpoint',required=True);p.add_argument('--scene',required=True);p.add_argument('--trajectory',required=True);p.add_argument('--output',required=True);p.add_argument('--start-s',type=float,default=3.2);train(**vars(p.parse_args()))
