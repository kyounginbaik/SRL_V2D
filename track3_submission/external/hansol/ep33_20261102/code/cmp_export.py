import numpy as np
from scipy.spatial.transform import Rotation as R
a=np.load('/data1/hansol/ep33_20261102/xfer/ep33_export_HGA0.npz'); b=np.load('/data1/hansol/ep33_20261102/xfer/ep33_export_HGA.npz')
A=a['world_from_object_rgb'][:,1]; B=b['world_from_object_rgb'][:,1]
ang=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('tji,tjk->tik',A[:,:3,:3],B[:,:3,:3])).as_rotvec(),axis=1))
pos=np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)*100
ch=np.where(ang>0.5)[0]; print('spoon frames differing >0.5deg:', len(ch), ch.min() if len(ch) else '-', ch.max() if len(ch) else '-', 'max deg', ang.max().round(1), 'max origin shift cm', pos.max().round(2))
