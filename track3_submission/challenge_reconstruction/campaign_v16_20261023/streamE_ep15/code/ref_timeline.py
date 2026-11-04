"""Fine timeline of an executed trajectory vs the episode-dir tracking reference for one object: per 0.25 s the reference
pose motion (speed, angular speed, rotation vs frame 0, height), the achieved error, the carrying-hand wrist speed, and contacts.
usage: ref_timeline.py TRAJ_NPZ SCENE_XML EPISODE_DIR OBJECT SIDE [T0 T1 STEP_FRAMES]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
traj,scene,epdir,obj,side=sys.argv[1:6]; t0=float(sys.argv[6]) if len(sys.argv)>6 else 0; t1=float(sys.argv[7]) if len(sys.argv)>7 else 1e9; st=int(sys.argv[8]) if len(sys.argv)>8 else 5
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); z=np.load(traj); fps=float(z['fps']); Q=z['qpos']; n=len(Q)
ref=TR.load_reference(Path(epdir)/'tracking_reference.npz',m)['qpos']; oc=int(m.joint(obj+'_free').qposadr[0])
ob=m.body(obj).id; palm=m.body(f'robot_{side}_{side}_base_yaw').id; ogeoms={g for g in range(m.ngeom) if m.geom_bodyid[g]==ob}
pr=ref[:,oc:oc+3]; Rr=R.from_quat(ref[:,oc+3:oc+7],scalar_first=True)
print('ref object start pos',np.round(pr[0],3),'end pos',np.round(pr[-1],3),'total rot start->end',round(np.degrees((Rr[-1]*Rr[0].inv()).magnitude()),1),'deg')
rv=(Rr*Rr[0].inv()).as_rotvec(); 
for k in range(0,n,st):
    if not (t0<=k/fps<=t1): continue
    d.qpos[:]=Q[k]; mujoco.mj_forward(m,d)
    p=d.xpos[ob]; Ra=R.from_matrix(d.xmat[ob].reshape(3,3)); e=np.linalg.norm(p-pr[k])*100; er=np.degrees((Ra*Rr[k].inv()).magnitude())
    k2=min(n-1,k+st); sp=np.linalg.norm(pr[k2]-pr[k])/((k2-k)/fps)*100 if k2>k else 0; asp=np.degrees((Rr[k2]*Rr[k].inv()).magnitude())/((k2-k)/fps) if k2>k else 0
    wp=d.xpos[palm]; d.qpos[:]=Q[k2]; mujoco.mj_forward(m,d); ws=np.linalg.norm(d.xpos[palm]-wp)/((k2-k)/fps)*100 if k2>k else 0
    d.qpos[:]=Q[k]; mujoco.mj_forward(m,d); cs={}
    for i in range(d.ncon):
        c=d.contact[i]; g1,g2=c.geom1,c.geom2
        for ga,gb in ((g1,g2),(g2,g1)):
            if ga in ogeoms:
                b=m.body(m.geom_bodyid[gb]).name; part='_'.join(b.split('_')[2:4]) if b.startswith('robot_') else b; cs[part]=cs.get(part,0)+1
    print(f'{k/fps:5.2f}s ref: z {pr[k][2]*100:5.1f}cm rot0 {np.degrees(np.linalg.norm(rv[k])):6.1f}deg v {sp:5.1f}cm/s w {asp:6.1f}deg/s | act err {e:5.2f}cm {er:6.1f}deg z {p[2]*100:5.1f} | wrist v {ws:5.1f} | {cs}')
