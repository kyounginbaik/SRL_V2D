import json,shutil,sys,numpy as np
from pathlib import Path
src=Path('/data1/hansol/ep33_20261102/orient/fit4/recon_E14_best_B2A_S1_HGAZr6/episode_000033')
tg=Path('/data1/hansol/retarget33/targets/R33_H5r16/episode_000033')
out=Path('/data1/hansol/ep33_20261102/agentmarks/bundle_H5r16/episode_000033')
if out.parent.exists(): shutil.rmtree(out.parent)
shutil.copytree(src,out)
z=dict(np.load(src/'object_poses.npz')); names=[str(n) for n in z['names']]; s=names.index('wooden_spoon'); c=names.index('blue_cup')
F=len(z['world_from_object'])
Cw=z['camera_from_object'][:,c]@np.linalg.inv(z['world_from_object'][:,c])
T=np.load(tg/'wooden_spoon_trajectory.npz')['T_world_obj']; print('F',F,'target',len(T))
T=T[:F]
print('r6 vs H5r16 spoon pos diff median cm', float(np.median(np.linalg.norm(T[:,:3,3]-z['world_from_object'][:len(T),s,:3,3],axis=1))*100))
z['world_from_object'][:len(T),s]=T; z['camera_from_object'][:len(T),s]=Cw[:len(T)]@T
np.savez_compressed(out/'object_poses.npz',**z)
mj=json.loads((out/'meshes.json').read_text()); shutil.copy2(tg/'wooden_spoon_mesh_metric.obj',out/mj['wooden_spoon']['file'])
print('ok')
