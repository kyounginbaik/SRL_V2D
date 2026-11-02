import sys,json,re,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
D=Path(sys.argv[1]);side=sys.argv[2];m=mujoco.MjModel.from_binary_path(str(D/'yam/yam_model.mjb'));d=mujoco.MjData(m);p=json.loads((D/'yam/yam_placement.json').read_text());up=np.array(p['up']);tab=p['table_height']
g=np.load(D/'baseline/physics.npz')['object_pose'][:,0,0,7:14];base=np.array(p['bases'][side]['pos']);other=np.array(p['bases']['left' if side=='right' else 'right']['pos'])
fwd=np.cross(up,other-base) if side=='left' else np.cross(other-base,up);fwd/=np.linalg.norm(fwd)
print('frames',len(g),'objects',p['objects'])
rel=g[:,:3]-base;rot=R.from_quat(g[:,3:7]);a0=rot[0]
for f in range(0,len(g),20):print(f,'t %.1f'%(f/20),'h_table %.1f cm'%(100*(g[f,:3]@up-tab)),'dist_base %.1f cm'%(100*np.linalg.norm(rel[f])),'forward %.1f'%(100*rel[f]@fwd),'turned %.0f deg'%np.degrees((rot[f]*a0.inv()).magnitude()))
w=(rot[1:]*rot[:-1].inv()).magnitude()*20;v=np.linalg.norm(np.diff(g[:,:3],axis=0),axis=1)*20
print('peak object speed %.2f m/s at %.1f s, peak turn %.1f rad/s at %.1f s'%(v.max(),v.argmax()/20,w.max(),w.argmax()/20))
print('dist_base max %.1f cm'%(100*np.linalg.norm(rel,axis=1).max()))
