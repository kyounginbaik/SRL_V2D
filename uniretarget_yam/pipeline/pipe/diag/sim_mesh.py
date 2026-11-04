"""One episode's objects in the simulator scene at frame 0: GT mesh on the left, collision parts on the right."""
import glob,json,os,sys
os.environ['MUJOCO_GL']='osmesa'
import numpy as np,mujoco,trimesh
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as RR
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';REG='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/';OUT='/scratch2/exaFLOPs/srl_yam_run/videos/mesh_sim/';FIT=json.load(open('/scratch2/exaFLOPs/srl_yam_run/source_meshes/table_fits.json'))
n=int(sys.argv[1]);mode=sys.argv[2];e=f'episode_{n:06d}';b=np.load(Z+f'bundles/{e}/object_poses.npz');names=[str(x) for x in b['names']];W=b['world_from_rgb'];fit=FIT[e];u=np.array(fit['up_world']);R1=RR.align_vectors([[0,0,1.]],[u])[0];f=R1.apply(W[:,:3,2]).mean(0);Rw=RR.from_euler('z',np.arctan2(f[0],f[1]))*R1;c0=Rw.apply(np.array(fit['center_world']))
s=mujoco.MjSpec();s.visual.global_.offwidth=960;s.visual.global_.offheight=720;s.visual.headlight.ambient[:]=.42;s.visual.headlight.diffuse[:]=.5;s.visual.headlight.specular[:]=.1
s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.225,.02,0],pos=[0,0,-.02],rgba=[.66,.38,.18,1])
PAL=[(.10,.70,.75,1),(.16,.40,.90,1)];info=[]
for k,nm in enumerate(names):
    T=np.load(Z+f'targets/{e}/{nm}_trajectory.npz')['T_world_obj'][0];gt=os.path.exists(REG+f'{e}/{nm}.obj');bd=s.worldbody.add_body(name=nm,pos=Rw.apply(T[:3,3])-c0,quat=(Rw*RR.from_matrix(T[:3,:3])).as_quat()[[3,0,1,2]])
    if mode=='gt':
        ms=[trimesh.load(REG+f'{e}/{nm}_decomp.obj' if gt else Z+f'targets/{e}/{nm}_mesh_metric.obj',force='mesh')];col=[PAL[k%2]]
    else:
        ps=sorted(glob.glob(f'/uniretarget_yam/objects/recon_{n:06d}/{nm}_collision/part_*.obj'));ms=[trimesh.load(p,force='mesh') for p in ps];rng=np.random.default_rng(k);col=[tuple(np.clip(np.array(PAL[k%2][:3])*rng.uniform(.65,1.25),0,1))+(1,) for _ in ms]
    info.append(f"{nm}: {'GT' if gt else 'no GT, reconstruction'}"+(f", {len(ms)} parts" if mode!='gt' else ''))
    for i,(m,cc) in enumerate(zip(ms,col)):
        s.add_mesh(name=f'{nm}_{i}',uservert=np.asarray(m.vertices).ravel().tolist(),userface=np.asarray(m.faces).ravel().tolist(),inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL);bd.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'{nm}_{i}',rgba=cc,density=0,mass=0)
m=s.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d);r=mujoco.Renderer(m,720,960);cam=mujoco.MjvCamera();cam.azimuth=-68;cam.elevation=-34;cam.distance=1.05;cam.lookat[:]=[0,0,.05];r.update_scene(d,cam);rgb=r.render().astype(float);seg=mujoco.Renderer(m,720,960);seg.enable_segmentation_rendering();seg.update_scene(d,cam);mask=seg.render()[...,0]<0
vy=np.linspace(0,1,720)[:,None,None];bg=(np.array([232,236,241.])*(1-vy)+np.array([186,192,200.])*vy)*np.ones((1,960,1));rgb[mask]=bg[mask];im=Image.fromarray(rgb.astype(np.uint8));dr=ImageDraw.Draw(im);dr.text((20,14),f"ep{n}  {'GT mesh' if mode=='gt' else 'collision parts'}",fill=(25,28,34),font_size=30)
for j,t in enumerate(info):dr.text((20,54+26*j),t,fill=(70,75,85),font_size=21)
im.save(OUT+f'.{mode}_{n}.png');print('ok',n,mode,flush=True)
