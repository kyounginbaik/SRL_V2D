"""Rotation error of an object vs the tracking reference, decomposed as a world-frame rotation vector (deg) and in the object's
reference frame; plus the world-frame position error vector. usage: rot_axis.py TRAJ_NPZ SCENE EPISODE_DIR OBJECT [--step 10]"""
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
traj,scene,epd,obj=sys.argv[1:5]; step=int(sys.argv[sys.argv.index('--step')+1]) if '--step' in sys.argv else 10
m=mujoco.MjModel.from_xml_path(scene); ref=TR.load_reference(Path(epd)/'tracking_reference.npz',m)['qpos']; tr=np.load(traj)
names=[str(x) for x in tr['object_names']]; i=names.index(obj); oc=int(m.joint(obj+'_free').qposadr[0]); n=min(len(ref),len(tr['qpos']))
p=tr['object_positions'][:n,i]; Ra=R.from_quat(tr['object_quaternions_wxyz'][:n,i][:,[1,2,3,0]]); pr=ref[:n,oc:oc+3]; Rr=R.from_quat(ref[:n,oc+3:oc+7],scalar_first=True)
E=(Ra*Rr.inv()); rv_w=np.degrees(E.as_rotvec()); rv_o=np.degrees((Rr.inv()*Ra).as_rotvec()); dp=(p-pr)*100
# spoon handle axis in world (reference): object x-axis assumed along the handle; print all three
ax=Rr.as_matrix()
print('t     |err| pos vec world(cm) x y z | rot world rotvec(deg) x y z |mag| | rot in ref-object frame x y z | ref obj axes (x,y,z) z-components')
for s in range(0,n,step):
    sl=slice(s,min(n,s+step)); r=rv_w[sl].mean(0); ro=rv_o[sl].mean(0); d=dp[sl].mean(0)
    print(f'{s/20:5.1f} {np.linalg.norm(d):5.2f} [{d[0]:5.2f} {d[1]:5.2f} {d[2]:5.2f}] | [{r[0]:6.1f} {r[1]:6.1f} {r[2]:6.1f}] {np.linalg.norm(r):5.1f} | [{ro[0]:6.1f} {ro[1]:6.1f} {ro[2]:6.1f}] | {ax[sl][:,2,0].mean():5.2f} {ax[sl][:,2,1].mean():5.2f} {ax[sl][:,2,2].mean():5.2f}')
