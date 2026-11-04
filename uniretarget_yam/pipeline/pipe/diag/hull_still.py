import json,os,sys
os.environ['MUJOCO_GL']='osmesa'
import numpy as np,mujoco
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as RR
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/bundles/'
OUT='/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/'
for spec in sys.argv[1:]:
    n,frames=spec.split(':');n=int(n);frames=[int(x) for x in frames.split(',')]
    e=f'/uniretarget_yam/episodes/recon_{n:06d}';M=json.load(open(e+'/recon_meta.json'));m=mujoco.MjModel.from_binary_path(e+'/yam/yam_model.mjb');d=mujoco.MjData(m);pad=int(M.get('pad',0))
    b=np.load(Z+f'episode_{n:06d}/object_poses.npz');K=json.load(open(Z+f'episode_{n:06d}/intrinsics.json'));names=[str(x) for x in b['names']];z=np.load(e+'/baseline/physics.npz');P=z['object_pose'][:,0]
    mv=max(M['movers'],key=lambda x:x['move_cm']);o=mv['object'];k=names.index(o);T0=b['world_from_object'][0,k];S0=np.eye(4);S0[:3,3]=P[0,k,7:10];S0[:3,:3]=RR.from_quat(P[0,k,10:14]).as_matrix();S=S0@np.linalg.inv(T0)
    robot=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_'))]
    for g in robot:m.geom_group[g]=5
    m.geom_group[m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE]=5
    pal=[(.95,.55,.1,1),(.1,.65,.9,1)]
    for i,nm in enumerate(names):
        G=np.flatnonzero(m.geom_bodyid==m.body(nm).id)
        for j,g in enumerate(G):
            sh=.75+.25*((j*7)%5)/4;m.geom_rgba[g]=(pal[i][0]*sh,pal[i][1]*sh,pal[i][2]*sh,1)
    m.vis.global_.fovy=float(np.degrees(2*np.arctan(.5*K['height']/K['fy'])));m.vis.global_.offwidth=K['width'];m.vis.global_.offheight=K['height'];m.vis.map.znear=.01;m.vis.headlight.ambient[:]=.5;m.vis.headlight.diffuse[:]=.5
    r=mujoco.Renderer(m,K['height'],K['width']);cam=mujoco.MjvCamera();adr={m.body(int(m.jnt_bodyid[j])).name:int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j]==0}
    for f in frames:
        d.qpos[:]=m.qpos0
        for i,nm in enumerate(names):
            if i<P.shape[1]:
                d.qpos[adr[nm]:adr[nm]+3]=P[f+pad,i,7:10];d.qpos[adr[nm]+3:adr[nm]+7]=P[f+pad,i,10:14][[3,0,1,2]]
        mujoco.mj_forward(m,d);C=S@b['world_from_rgb'][f];r.update_scene(d,cam)
        for c in r.scene.camera:c.pos[:]=C[:3,3];c.forward[:]=C[:3,2];c.up[:]=-C[:3,1]
        Image.fromarray(r.render()).save(OUT+f'.hull_{n}_{f}.png');print('saved',n,f,flush=True)
