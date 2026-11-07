"""Zero-residual (pure reference tracking) rollout. Otherwise identical to the official CHORD evaluation (VOC 0, deterministic, from frame 0) of one checkpoint, also recording the rollout for scoring and video."""
import sys,json,numpy as np
sys.path.insert(0,'/workspace/src')
ckpt,out=sys.argv[1:3]; W=int(sys.argv[3]) if len(sys.argv)>3 else 32
from flash_chord.envs import rl as RL
from flash_chord.evaluation import harness as H
rec=dict(q=[],b=[],o=[]); env_ref={}
cls=[c for c in vars(RL).values() if isinstance(c,type) and c.__module__==RL.__name__ and hasattr(c,'step') and hasattr(c,'reset')][0]
_step=cls.step
def step(self,*a,**k):
    self.action_input.zero_(); r=_step(self,None); env_ref['e']=self
    n=self.world_count; rec['q'].append(self.state_0.joint_q.numpy().reshape(n,-1)[:4].copy()); bq=self.state_0.body_q.numpy().reshape(n,-1,7); rec['b'].append(bq[:4].copy())
    ids=self.command.body_ids_w.numpy().reshape(-1); rec['o'].append(self.state_0.body_q.numpy()[ids].reshape(n,-1,7).copy())
    return r
cls.step=step
oc=H.evaluate_checkpoint(ckpt,world_count=W,start_frame=0,reset_mode='explicit')
e=env_ref['e']; m=e.model
try: jl=list(m.joint_label); bl=list(m.body_label)
except Exception: jl=[];bl=[]
jq=np.asarray(m.joint_q_start.numpy()) if hasattr(m,'joint_q_start') else np.array([])
ref=H.evaluation_scalars(oc) if hasattr(H,'evaluation_scalars') else {}
from flash_chord.evaluation.harness import evaluation_scalars
sc=evaluation_scalars(oc)
np.savez_compressed(out,q=np.asarray(rec['q'],np.float32),b=np.asarray(rec['b'],np.float32),o=np.asarray(rec['o'],np.float32),joint_labels=np.array(jl),body_labels=np.array(bl),joint_q_start=jq,scalars=json.dumps(sc,default=float),step_count=oc.step_count)
print('EVAL',json.dumps({k:v for k,v in sc.items()},default=float)[:1500]); print('FRAMES',len(rec['q']),'W',W,'first_bodies',bl[:6],'J',jl[:9],'stepcount',oc.step_count)
