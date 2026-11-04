"""Per-second object error timeline of an executed trajectory vs the episode dir's tracking reference, plus wrist-object distance.
usage: err_timeline.py TRAJECTORY_NPZ EPISODE_DIR SCENE_XML [--step S]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
traj, epdir, scene = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
step=int(sys.argv[sys.argv.index('--step')+1]) if '--step' in sys.argv else 20
m=mujoco.MjModel.from_xml_path(scene); ref=TR.load_reference(epdir/'tracking_reference.npz', m)['qpos']; tr=np.load(traj)
names=[str(x) for x in tr['object_names']]; fps=20.; n=min(len(ref),len(tr['qpos']))
d=mujoco.MjData(m)
def wrist_pos(q):
    d.qpos[:]=q; mujoco.mj_kinematics(m,d)
    return {s:d.body(f'robot_{s}_{s}_base_yaw').xpos.copy() for s in ('left','right')}
W=[wrist_pos(tr['qpos'][k]) for k in range(n)]
for i,o in enumerate(names):
    oc=int(m.joint(o+'_free').qposadr[0]); p_ref=ref[:n,oc:oc+3]; R_ref=R.from_quat(ref[:n,oc+3:oc+7],scalar_first=True)
    p=tr['object_positions'][:n,i]; Ra=R.from_quat(tr['object_quaternions_wxyz'][:n,i][:,[1,2,3,0]])
    e=np.linalg.norm(p-p_ref,axis=1)*100; er=np.degrees((Ra*R_ref.inv()).magnitude())
    dl=np.array([np.linalg.norm(W[k]['left']-p[k]) for k in range(n)]); dr=np.array([np.linalg.norm(W[k]['right']-p[k]) for k in range(n)])
    print(f'{o}: mean {e.mean():.2f} cm / {er.mean():.1f} deg  (frames {n})')
    for s in range(0,n,step):
        sl=slice(s,min(n,s+step)); sp=np.linalg.norm(np.diff(p_ref[sl],axis=0),axis=1).mean()*fps*100 if sl.stop-sl.start>1 else 0
        print(f'  {s/fps:5.1f}s  pos {e[sl].mean():5.2f} cm  rot {er[sl].mean():5.1f} deg | hand-obj dist L {dl[sl].mean()*100:5.1f}±{dl[sl].std()*100:4.1f} R {dr[sl].mean()*100:5.1f}±{dr[sl].std()*100:4.1f} cm | ref speed {sp:4.1f} cm/s')
