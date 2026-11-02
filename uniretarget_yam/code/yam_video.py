"""Roll out YAM samples in physics and save the state the renderer reads; one world per round."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from yam_loop import YamTask
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=5);a=p.parse_args()
t=YamTask(a.episode,a.side,a.object);t.set_stage('real');t.nominal_physics();items=[]
for f in sorted(a.run.glob('milestones/frontier_*.json')):items.append((f.stem,json.loads(f.read_text())['result']['sample']))
items=items[:1]+items[1:][-(a.limit-2):] if len(items)>a.limit-1 else items
for name in ('pass.json','top.json'):
    f=a.run/name
    if f.exists():
        r=json.loads(f.read_text());r=r[0] if isinstance(r,list) else r;items.append(('final_'+name.split('.')[0],r['sample']));break
base=t.poses;H=t.frames-1;a.output.mkdir(parents=True,exist_ok=True);names=[]
t.stage='video'
for w,(name,s) in enumerate(items):
    C,_=t.compile(s,H);t.stage='real';saved=t.spent if hasattr(t,'spent') else None
    m=t.m;r=None
    # Record the whole episode: stopping rules are for search, a video shows what really happens.
    import mujoco
    d=t.d;mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d);Q=[]
    for f in range(min(H,len(C))):
        d.ctrl[:]=C[f]
        for _ in range(20):mujoco.mj_step(m,d)
        Q.append(d.qpos.copy())
    Q=np.array(Q);P=np.zeros((len(Q),1,base.shape[1],14))
    for k,qi in enumerate(t.object_q):P[:,0,k,:3]=Q[:,qi:qi+3];P[:,0,k,3:7]=Q[:,qi+3:qi+7][:,[1,2,3,0]];P[:,0,k,7:14]=base[:len(Q),k,7:14]
    d0=a.output/f'r{w}';d0.mkdir(exist_ok=True);np.savez_compressed(d0/'physics.npz',native_qpos=Q[:,None],object_pose=P,native_capture_world_ids=np.array([0]),fps=np.array(20.));o=P[:,0,a.object];ep=np.linalg.norm(Q[:,t.qo:t.qo+3]-t.gp_raw[:len(Q)],axis=1);print(json.dumps(dict(round=w,name=name,mean_cm=round(float(100*ep.mean()),2),lift_mm=round(float(1e3*((Q[:,t.qo:t.qo+3]-t.o0)@t.up).max()),1),cause=int(s.get('cause',-1)) if 'cause' in s else -1)),flush=True);names.append(name);t.stage='video'
(a.output/'rounds.json').write_text(json.dumps(names))
