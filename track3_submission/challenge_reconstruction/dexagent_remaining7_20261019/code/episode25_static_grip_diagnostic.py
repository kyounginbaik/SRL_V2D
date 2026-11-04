"""Counterfactual acquisition-free lift test; NEVER a submission candidate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from scipy.spatial.transform import Rotation
from refine_episode25_geometry import W,B,Experiment,H,G,servo_rollout,dump

root=W/'refinement/episode_000025/peg_v1';spec=json.loads((root/'spec.json').read_text());exp=Experiment(spec);m=exp.model;cols=exp.cols['left'];oc=exp.objects['wooden_piece_1'];rig=H.HandRig.build(m,'robot_left_','left')
settled=np.load(root/'passive_forecast.npz')['qpos'][-1];sR=Rotation.from_quat(settled[oc+3:oc+7],scalar_first=True).as_matrix();reports=[]
for seedid in [0,36,72,108]:
 seed=np.load(root/'proposals'/f'{seedid:04d}.npz')['qpos'];anchor=60
 r=Rotation.from_quat(seed[anchor,oc+3:oc+7],scalar_first=True).as_matrix();localp=r.T@(seed[anchor,cols[:3]]-seed[anchor,oc:oc+3]);localR=r.T@H.euler_to_mat(seed[anchor,cols[3:6]])
 q=settled.copy();q[cols[:3]]=settled[oc:oc+3]+sR@localp;q[cols[3:6]]=G.nearest_euler(H.mat_to_euler(sR@localR),seed[anchor,cols[3:6]]);q[cols[6:]]=seed[anchor,cols[6:]]
 for pressure in [0.,.15,.3,.6]:
  Q=np.repeat(q[None],80,axis=0)
  for j,name in enumerate(rig.finger_qpos):
   if any(k in name for k in ['MCP_FE','PIP','DIP','thumb_IP']):Q[10:,cols[6+j]]+=pressure
  lift=np.clip((np.arange(80)-20)/30,0,1)*.10;Q[:,cols[2]]+=lift
  Q[:,cols]=np.clip(Q[:,cols],exp.limits['left'][:,0],exp.limits['left'][:,1])
  roll=servo_rollout(m,Q,20,qpos_start=q,record_substeps=False)
  height=roll['qpos'][:,oc+2]-q[oc+2];error=np.linalg.norm((roll['qpos'][:,oc:oc+3]-q[oc:oc+3])-np.c_[np.zeros((80,2)),lift],axis=1)
  row={'seed':seedid,'pressure':pressure,'max_lift_m':float(height.max()),'final_lift_m':float(height[-1]),'final_position_error_m':float(error[-1]),'max_hand_pen_m':float(roll['hand_pen_m'].max()),'NOT_SUBMITTABLE':True,'different_initial_robot_state':True,'passively_settled_initial_objects':True};reports.append(row);print(json.dumps(row),flush=True)
  np.savez_compressed(root/f'diagnostic_seed{seedid}_press{pressure}.npz',**{k:roll[k] for k in ['qpos','qvel','ctrl','time_s','hand_pen_m','hand_object_force_n']})
dump(B/'review/episode25_static_grip_diagnostic.json',reports)
