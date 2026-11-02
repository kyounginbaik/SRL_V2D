"""Refine one episode end to end: find moving objects, assign the nearest hand, refine in turn."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from fastsim import FastSim
p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--tag',default='auto');p.add_argument('--seed',type=int,default=21);p.add_argument('--stall',type=int,default=30);p.add_argument('--generations',type=int,default=200);p.add_argument('--no-ladder',action='store_true');p.add_argument('--adaptive',action='store_true');a=p.parse_args()
model=a.episode/'native_model.mjb';physics=a.episode/'baseline/physics.npz';plan=a.episode/'plan.npz';s=FastSim(model,physics);m=s.m;d=s.d;poses=s.poses;order=[]
with np.load(plan) as z:Q=z['joint_q'][0]
wrists={side:[j for j in range(m.njnt) if m.jnt_type[j]==1 and m.body(m.jnt_bodyid[j]).name.startswith(side+'_')][0] for side in ('left','right')}
for k in range(poses.shape[1]):
    o=poses[:,k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);moved=(np.linalg.norm(gp-gp[0],axis=1)>.02)|((gr*gr[0].inv()).magnitude()>.35)
    if not moved.any():continue
    onset=int(np.flatnonzero(moved)[0]);q=s.native(Q[min(onset,len(Q)-1)]);d.qpos[:]=q;mujoco.mj_kinematics(m,d);dist={side:float(np.linalg.norm(d.xpos[m.jnt_bodyid[j]]-gp[onset])) for side,j in wrists.items()};order.append((onset,k,min(dist,key=dist.get),dist))
order.sort();print(json.dumps(dict(moving=[dict(onset=o,object=k,side=h,wrist_distance={x:round(v,3) for x,v in di.items()}) for o,k,h,di in order],objects=poses.shape[1])),flush=True)
used=set(h for _,_,h,_ in order);current=plan;failed=[]
for step,(onset,k,side,_) in enumerate(order):
    out=a.episode/f'{a.tag}_step{step}_obj{k}'
    cmd=[sys.executable,str(Path(__file__).parent/'frontier_loop.py'),'--model',str(model),'--physics',str(physics),'--plan',str(current),'--output',str(out),'--object',str(k),'--side',side,'--workers',str(a.workers),'--population',str(10*a.workers),'--proposals',str(40*a.workers),'--seed',str(a.seed+step),'--stall',str(a.stall),'--generations',str(a.generations),'--manifest',str(a.episode/'manifest.json'),'--future']+[str(x[1]) for x in order[step+1:]]+['--pending']+[str(x) for x in failed]+(['--park-other'] if any(x[2]!=side for x in order[step+1:]) else [])+(['--no-ladder'] if a.no_ladder else [])+(['--adaptive'] if a.adaptive else [])+(['--no-park'] if len(order)>1 else [])
    subprocess.run(cmd,check=True);current=out/'best_plan.npz'
    # A failed object must not cut off the next one; the final verdict still counts it.
    if not (out/'canonical_pass.json').exists():failed.append(k)
    print(json.dumps(dict(finished_step=step,object=k,side=side,plan=str(current))),flush=True)
# Static objects also need judging, so report the official verdict from the last step's gate only.
final=a.episode/f'{a.tag}_final';final.mkdir(exist_ok=True);(final/'plan.npz').write_bytes(Path(current).read_bytes());(final/'manifest.json').write_text((a.episode/'manifest.json').read_text());print(json.dumps(dict(final=str(final))),flush=True)
# The verdict that counts: canonical physics on the whole episode, every object, from the real reset.
subprocess.run([sys.executable,str(Path(__file__).parent/'evaluate_native.py'),'--plan',str(final),'--output',str(final/'verdict'),'--replicas','1'],check=True)
result=json.loads((final/'verdict/result.json').read_text());steps=[dict(object=k,side=h,passed=(a.episode/f'{a.tag}_step{i}_obj{k}/canonical_pass.json').exists()) for i,(_,k,h,_) in enumerate(order)]
print(json.dumps(dict(episode_verdict=a.episode.name,official_mp_sr=result['official_mp_sr'],full_episode=result['full_episode'],objects_refined=steps)),flush=True)
