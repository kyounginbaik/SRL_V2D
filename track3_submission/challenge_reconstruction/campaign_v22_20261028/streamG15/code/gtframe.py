import numpy as np, kitx, json
from scipy.spatial.transform import Rotation as Rot
for ep in kitx.TUNE+kitx.HELD:
    X=kitx.load_export(ep,'r8_base'); z=np.load(kitx.root(ep)/'exports/r8_base/object_poses.npz')
    for i,n in enumerate(X['names']):
        G=kitx.K.gt_world_T_obj(ep,n)[0]; C=z['camera_from_object'][0,i]
        print(ep,n, 'gt f0',np.round(G[0,:3,3],3),'ours cam f0',np.round(C[:3,3],3), 'gt |p|',round(np.linalg.norm(G[0,:3,3]),3),'ours |p|',round(np.linalg.norm(C[:3,3]),3))
