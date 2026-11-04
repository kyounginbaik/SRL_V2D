"""Summarise the pot-mass grid: proxy pass, per-object errors, pot tilt/sag while lifted, and 200 g control identity."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
root=Path(__file__).resolve().parents[1]; grid=root/'refinement_20261020_pot/mass_grid'
rows=[]
for e in (3,5,6,8):
    sub=pd.read_parquet(root/f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/results/episode_{e:06d}.parquet')
    p=sub[sub.object_body_name=='white_pot'].sort_values('step')
    qr=R.from_quat(p[['reference_quaternion_x','reference_quaternion_y','reference_quaternion_z','reference_quaternion_w']].to_numpy())
    pr=p[['reference_position_x','reference_position_y','reference_position_z']].to_numpy()
    lifted=pr[:,2]>pr[0,2]+0.03; up=qr[0].inv().apply([0,0,1])
    orig=json.load(open(root/f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/portable/original/evaluation_result.json'))
    orig_traj=np.load(orig['execution']['trajectory'])
    for d in sorted(grid.glob(f'episode_{e:06d}/mass_*')):
        r=d/'result.json'
        if not r.exists(): rows.append(dict(ep=e,mass_g=int(d.name[5:8]),status='MISSING')); continue
        res=json.load(open(r)); m=res['metrics']['official_style_reference_proxy']; tr=np.load(res['execution']['trajectory'])
        names=list(tr['object_names']); i=names.index('white_pot')
        qa=tr['object_quaternions_wxyz'][:,i][:,[1,2,3,0]]; Ra=R.from_quat(qa)
        # trajectory positions are in the sim frame; reference parquet is exported frame -> compare orientation only (frame is a pure z-shift)
        tilt=np.degrees(np.arccos(np.clip(np.sum(Ra.apply(up)*qr.apply(up),1),-1,1)))
        pa=tr['object_positions'][:,i]; dz=(pa[:,2]-pa[0,2])-(pr[:,2]-pr[0,2])
        objs={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()}
        same=bool(np.array_equal(tr['qpos'],orig_traj['qpos'])) if d.name=='mass_200g' else None
        rows.append(dict(ep=e,mass_g=int(d.name[5:8]),status='PASS' if m['mp_sr']==1 else 'fail',
            pot=objs['white_pot'],other={k:v for k,v in objs.items() if k!='white_pot'},
            tilt_lifted_deg=round(float(tilt[lifted].mean()),1),tilt_max_deg=round(float(tilt.max()),1),
            sag_lifted_cm=round(float(dz[lifted].mean()*100),2),
            completed=res['execution']['execution_completed'],replay_ok=res['execution']['independent_control_replay']['ok'],
            identical_to_submitted=same))
for r in rows: print(r)
json.dump(rows,open(root/'refinement_20261020_pot/mass_grid_summary.json','w'),indent=1,default=str)
