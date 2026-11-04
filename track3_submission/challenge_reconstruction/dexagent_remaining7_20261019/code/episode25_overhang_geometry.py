"""Planning-only reachability audit; object translations here are not rollouts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,numpy as np
from episode25_pinch_transfer import W,B,Experiment,pinch,poses,dump
out=W/'review/episode_000025/overhang_geometry';out.mkdir(parents=True,exist_ok=True)
spec=json.loads((W/'refinement/episode_000025/peg_v1/spec.json').read_text());exp=Experiment(spec)
contacts,actual,closed,opened=pinch();settled=np.load(W/'refinement/episode_000025/peg_v1/passive_forecast.npz')['qpos'][-1];oc=exp.objects['wooden_piece_2'];results=[]
for offset in [.025,.05,.075]:
 q=settled.copy();q[oc]+=offset;p=out/f'offset_{offset:g}';p.mkdir(exist_ok=True);c=poses(exp,'right',q,contacts,actual,closed,opened,p)
 row={'hypothetical_world_x_translation_m':offset,'feasible_grasps':len(c),'planning_only':True,'not_a_physical_execution':True};results.append(row);print(json.dumps(row),flush=True)
 if c:
  np.savez_compressed(p/'grasp_seeds.npz',local_positions=np.array([v[1] for v in c]),local_rotations=np.array([v[2] for v in c]),marker_centers=np.array([v[3] for v in c]),closed=closed,opened=opened,actual=actual,planning_object_pose=q[oc:oc+7])
dump(out/'summary.json',results)
