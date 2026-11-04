#!/usr/bin/env python3
"""Read recorded rack/hand contact geometry; does not alter a simulation run."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE,ROOT,LEGACY,write
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from tool_library.common import hand as H

run=BASE/'physics_collision_v2/sweeper_transfer/donor3_park0_v1/pickup_search_v1'
result=json.loads((run/'result.json').read_text());model=mujoco.MjModel.from_xml_path(result['execution']['scene'])
record=np.load(result['execution']['trajectory']);ref=np.load(BASE/'physics_collision_v2/episodes/episode_000030/tracking_reference.npz')['qpos']
rack=model.body('plastic_dish_rack').id;adr=int(model.joint('plastic_dish_rack_free').qposadr[0]);fps=float(record['fps'])
rigs={s:H.HandRig.build(model,f'robot_{s}_',s) for s in ['left','right']}
rows=[]
for seconds in [0.,1.,1.8,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.,4.,6.,8.,10.,12.]:
    k=round(seconds*fps);d=mujoco.MjData(model);d.qpos[:]=record['qpos'][k];d.qvel[:]=record['qvel'][k];d.ctrl[:]=record['ctrl'][k];mujoco.mj_forward(model,d)
    contacts=[];force=np.zeros(6)
    for i,c in enumerate(d.contact):
        b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);other=b2 if b1==rack else b1 if b2==rack else -1
        if other<0 or not model.body(other).name.startswith('robot_'):continue
        mujoco.mj_contactForce(model,d,i,force)
        contacts.append(dict(body=model.body(other).name,force_n=float(force[0]),penetration_mm=max(0.,-float(c.dist)*1000),
            contact_body_m=((c.pos-d.xpos[rack])@d.xmat[rack].reshape(3,3)).tolist()))
    tips={s:{f:((d.site_xpos[rig.fingertip_sites[f]]-d.xpos[rack])@d.xmat[rack].reshape(3,3)).tolist() for f in H.FINGERS} for s,rig in rigs.items()}
    rotation=(Rotation.from_quat(ref[k,adr+3:adr+7],scalar_first=True).inv()*Rotation.from_quat(d.qpos[adr+3:adr+7],scalar_first=True)).magnitude()
    rows.append(dict(seconds=seconds,rack_error_cm=float(np.linalg.norm(d.qpos[adr:adr+3]-ref[k,adr:adr+3])*100),rack_error_deg=float(np.degrees(rotation)),contacts=contacts,fingertips_rack_body_m=tips))
output=BASE/'review/episode30_rack_contact_timeline.json';write(output,dict(recording=result['execution']['trajectory'],rows=rows))
for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='fingertips_rack_body_m'}),flush=True)
