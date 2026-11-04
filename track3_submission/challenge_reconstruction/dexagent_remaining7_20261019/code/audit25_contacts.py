import sys,json
from pathlib import Path
import numpy as np,mujoco
from refine_episode25_geometry import B,W,E,H,Experiment
out=W/'refinement/episode_000025/pickup_left_v1'
spec=json.loads((out/'spec.json').read_text());exp=Experiment(spec);m=exp.model;d=mujoco.MjData(m);r=np.load(out/'execution/trajectory.npz');result=json.loads((out/'result.json').read_text());Q=np.load(result['reference'])['qpos'];rig=H.HandRig.build(m,'robot_left_','left');bid=m.body('wooden_piece_1').id;sites=list(rig.fingertip_sites.values());report=[]
for t in [0,20,30,38,40,42,44,46,48,50,52,54,56,60,70,80]:
 row={'frame':t,'recorded_force':r['hand_object_force_n'][t].tolist(),'object_xyz':r['qpos'][t,56:59].tolist(),'desired_wrist':Q[t,:6].tolist(),'actual_wrist':r['qpos'][t,:6].tolist()}
 for label,qp in [('actual',r['qpos'][t]),('command',Q[t])]:
  d.qpos[:]=qp
  if label=='command':d.qpos[56:]=r['qpos'][t,56:]
  mujoco.mj_forward(m,d)
  row[label+'_tips_in_object']=((d.site_xpos[sites]-d.xpos[bid])@d.xmat[bid].reshape(3,3)).tolist();contacts=[]
  for c in d.contact:
   b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
   if bid not in [b1,b2] and m.body(b1).name!='table' and m.body(b2).name!='table':continue
   if b1 not in rig.body_ids and b2 not in rig.body_ids:continue
   contacts.append({'bodies':[m.body(b1).name,m.body(b2).name],'depth_mm':-float(c.dist)*1000})
  row[label+'_contacts']=contacts
 report.append(row)
(B/'review/episode25_pickup_contact_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
