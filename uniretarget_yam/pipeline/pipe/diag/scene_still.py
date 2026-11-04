import json,os,sys
os.environ['MUJOCO_GL']='egl'
import numpy as np,mujoco
from PIL import Image
from scipy.spatial.transform import Rotation as R
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/bundles/'
for n in map(int,sys.argv[1:]):
    e=f'/uniretarget_yam/episodes/recon_{n:06d}';M=json.load(open(e+'/recon_meta.json'));m=mujoco.MjModel.from_binary_path(e+'/yam/yam_model_visual.mjb');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    b=np.load(Z+f'episode_{n:06d}/object_poses.npz');K=json.load(open(Z+f'episode_{n:06d}/intrinsics.json'));names=[str(x) for x in b['names']];o=M['objects'][0]['name'];bid=m.body(o).id;T=b['world_from_object'][0,names.index(o)]
    Sim=np.eye(4);Sim[:3,:3]=d.xmat[bid].reshape(3,3);Sim[:3,3]=d.xpos[bid];S=Sim@np.linalg.inv(T);C=S@b['world_from_rgb'][0]
    m.vis.global_.fovy=float(np.degrees(2*np.arctan(.5*K['height']/K['fy'])));m.vis.global_.offwidth=K['width'];m.vis.global_.offheight=K['height'];m.vis.map.znear=.01
    r=mujoco.Renderer(m,K['height'],K['width']);cam=mujoco.MjvCamera();r.update_scene(d,cam)
    for c in r.scene.camera:c.pos[:]=C[:3,3];c.forward[:]=C[:3,2];c.up[:]=-C[:3,1]
    Image.fromarray(r.render()).save(f'/uniretarget_yam/still_{n}.png');print('STILL',n,[m.geom(g).name for g in range(m.ngeom) if m.geom_bodyid[g]==0],'support',m.body('support').pos.round(3),m.geom('support_top').size[:2],flush=True)
