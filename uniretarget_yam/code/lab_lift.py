"""Gentle-lift ablation of placed grasps: separates holding from fast-motion failure."""
import sys,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab
root=Path(__file__).parent.parent/'runs'
real=Lab(root/'surface_anchor_fit_v6/native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',19)
z0=np.load(root/'surface_anchor_fit_v6/candidate_015.npz')['qpos'];op=z0[real.qo:real.qo+3].copy();or0=R.from_quat(z0[real.qo+3:real.qo+7][[1,2,3,0]])
com=op+or0.apply(real.m.body_ipos[real.obj]);print('COM',np.round(com,3),'object origin',np.round(op,3))
def synthetic(height,seconds,turn_deg=0.):
    n=len(real.gp);u=np.clip((np.arange(n)-real.onset)/(seconds*20),0,1);s=u*u*(3-2*u)
    gp=op+np.outer(s*height,real.up);ax=np.array([1.,0,0]);gr=R.from_rotvec(np.outer(s*np.radians(turn_deg),ax))*or0
    return gp,gr
for cname in ['candidate_015','candidate_009','candidate_004','candidate_010','candidate_007']:
    z=np.load(root/f'surface_anchor_fit_v6/{cname}.npz');P,N,G=z['patch_positions'],z['patch_normals'],z['hand_geom_ids']
    print(cname,'contact-to-COM cm',np.round(100*np.linalg.norm(P-com,axis=1),1).tolist())
    for label,(h,s,t) in dict(lift_slow_8cm=(.08,2.,0.),lift_fast_8cm=(.08,.5,0.),turn_slow=(.08,2.,60.)).items():
        gp,gr=synthetic(h,s,t);real.gp=gp;real.gr=gr;real.gp_raw=gp;real.gr_raw=gr
        for grip in (1.,3.,8.,20.):
            r=real.run(z['qpos'],P,N,G,grip,horizon=int(s*20)+20)
            print(f"  {label:14s} grip{grip:4.0f} lift_mm {r.get('lift_mm',0):6.1f} mean_cm {r.get('mean_cm',0):5.1f} deg {r.get('mean_deg',0):5.1f} within {r.get('within',0):.2f} contact {r.get('contact_frames',0)}")
