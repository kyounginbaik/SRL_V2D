"""Per-candidate physical summary: tracking, contact, lift, first official-tolerance break."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.native_contact_analysis import object_contact_snapshot
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--start',type=int,default=15);p.add_argument('--end',type=int,default=120);a=p.parse_args()
d=np.load(a.run/'physics.npz');r=np.load(a.run/'references.npz');res=json.loads((a.run/'result.json').read_text())
names=[c['name'] for c in json.loads((a.plan/'manifest.json').read_text())['candidates']] if a.plan else None
g=d['world_reference_group'];cap=d['native_capture_world_ids'];pose=d['object_pose'];n=pose.shape[0];qs=d['newton_joint_q_start']
grav=d['newton_gravity'][0];up=-grav/np.linalg.norm(grav);obj=int(d['native_object_body_ids'][0,0]);mg=float(d['native_model_body_mass'][53]*np.linalg.norm(grav))
Q=d['native_qpos'];print('candidate name first3cm first30deg firstAny maxlift_mm maxErrUntil100_cm wristTrackR_cm contactFrames maxHandUp/mg')
end=min(a.end,n)
for i,c in enumerate(cap):
    grp=int(g[c]);o=pose[:,c,0];init=o[0];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();off=init[:3]-al.apply(init[7:10])
    gp=al.apply(o[:,7:10])+off;gr=al*R.from_quat(o[:,10:14]);ep=np.linalg.norm(o[:,:3]-gp,axis=1);er=(R.from_quat(o[:,3:7]).inv()*gr).magnitude()
    f3=np.flatnonzero(ep>.03);fr=np.flatnonzero(er>np.radians(30));first=lambda x:int(x[0]) if len(x) else -1
    anyf=[x for x in (first(f3),first(fr)) if x>=0]
    lift=((o[:,:3]-o[0,:3])@up).max()*1e3
    # Right-hand wrist native qpos 29..31 compared with the reference (newton 29..31).
    w=np.linalg.norm(Q[:,i,29:32]-r['joint_q'][grp][:n,29:32],axis=1)[a.start:end].mean()*100
    contact=0;load=0.
    for f in range(a.start,end):
        s=object_contact_snapshot(d,f,int(c));mask=(s.counterpart_newton_bodies>=0)&(s.counterpart_newton_bodies<int(d['native_object_body_ids'][i,0]))
        if mask.any():contact+=1;load=max(load,float(s.forces_on_object[mask].sum(0)@up/mg))
    print(grp,(names[grp] if names else ''),first(f3),first(fr),min(anyf) if anyf else -1,round(lift,1),round(ep[:min(100,n)].max()*100,1),round(float(w),2),contact,round(load,2))
