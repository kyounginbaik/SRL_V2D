from pathlib import Path
import numpy as np,mujoco,json,sys
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017');sys.path.insert(0,str(b.parents[1]/'third_party/DexAgent'));from tool_library.common import hand as H
p=b/'refinement_20261017/lid_seed/episode_000001/candidate_00/execution';m=mujoco.MjModel.from_xml_path(str(p/'scene_physics.xml'));d=mujoco.MjData(m);r=dict(np.load(p/'trajectory.npz'));rig=H.HandRig.build(m,'robot_right_','right');l=m.body('white_pot_lid').id
cols=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]+list(rig.finger_qpos.values()));out=[]
for t in [1.2,1.4,1.6,1.7,1.8,2.,2.2,2.4,2.6,3.]:
 f=round(t*20);row={'time_s':t}
 for mode in ['actual','commanded_hand_actual_objects']:
  d.qpos[:]=r['qpos'][f]
  if mode!='actual':d.qpos[cols]=r['reference'][f,cols]
  mujoco.mj_forward(m,d);contacts=[]
  for c in d.contact:
   a,z=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);h=z if a==l else a if z==l else -1
   if h in rig.body_ids:contacts.append({'body':m.body(h).name,'penetration_m':max(0.,-float(c.dist))})
  row[mode]=contacts
 row['wrist_tracking_error_m']=float(np.linalg.norm(r['qpos'][f,cols[:3]]-r['reference'][f,cols[:3]]))
 row['finger_tracking_error_deg']=float(np.rad2deg(np.abs(r['qpos'][f,cols[6:]]-r['reference'][f,cols[6:]])).mean())
 out.append(row);print(row)
(b/'refinement_20261017/lid_seed/episode_000001/candidate_00/contact_probe.json').write_text(json.dumps(out,indent=2))
