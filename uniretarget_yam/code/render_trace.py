"""Annotated physics video: goal ghost, trace cause, tolerance break, per-round timeline."""
import argparse,json,os
os.environ.setdefault('MUJOCO_GL','osmesa')
from pathlib import Path
import numpy as np,mujoco,cv2,imageio
from scipy.spatial.transform import Rotation as R
def loop_frontier(ep,er,total):
    # The loop's own frontier: object lost, or the official mean can no longer be met.
    n=np.arange(1,len(ep)+1);limit=.7*n+.3*total
    over=np.flatnonzero((ep>.15)|(er>np.radians(150.))|(np.cumsum(ep)>.03*limit)|(np.cumsum(er)>np.radians(30.)*limit))
    return int(over[0]) if len(over) else -1
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=0);p.add_argument('--title',default='');p.add_argument('--cause',type=int,default=-1);p.add_argument('--output',type=Path,required=True);p.add_argument('--end',type=int);p.add_argument('--slow',type=int,default=2);p.add_argument('--object',type=int,default=0);a=p.parse_args()
m=mujoco.MjModel.from_binary_path(str(a.model));m.vis.global_.offwidth=720;m.vis.global_.offheight=540;d=mujoco.MjData(m);r=mujoco.Renderer(m,540,720)
z=np.load(a.run/'physics.npz');Q=z['native_qpos'][:,a.world];pose=z['object_pose'][:,int(z['native_capture_world_ids'][a.world]),a.object]
init=pose[0];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();off=init[:3]-al.apply(init[7:10])
gp=al.apply(pose[:,7:10])+off;gr=al*R.from_quat(pose[:,10:14]);ep=np.linalg.norm(pose[:,:3]-gp,axis=1);er=(R.from_quat(pose[:,3:7]).inv()*gr).magnitude()
first=loop_frontier(ep,er,len(ep)+1)
n=min(a.end or len(Q),len(Q));free=[j for j in range(m.njnt) if m.jnt_type[j]==0];obj=int(m.jnt_bodyid[min(free,key=lambda j:float(np.linalg.norm(z['native_qpos'][0,0][m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-z['object_pose'][0,0,a.object,:3])))]);og=np.flatnonzero(m.geom_bodyid==obj);cam=mujoco.MjvCamera();cam.distance=.75;cam.elevation=-28;cam.azimuth=115;cam.lookat[:]=gp[:n].mean(0)
fps=float(z['fps']);frames=[];others={}
for k_ in range(z['object_pose'].shape[2]):
    if k_==a.object:continue
    o_=z['object_pose'][:,int(z['native_capture_world_ids'][a.world]),k_];i_=o_[0];al_=R.from_quat(i_[3:7])*R.from_quat(i_[10:14]).inv();gp_=al_.apply(o_[:,7:10])+i_[:3]-al_.apply(i_[7:10]);gr_=al_*R.from_quat(o_[:,10:14]);e1_=np.linalg.norm(o_[:,:3]-gp_,axis=1);e2_=(R.from_quat(o_[:,3:7]).inv()*gr_).magnitude();others[k_]=(e1_,e2_,loop_frontier(e1_,e2_,len(e1_)+1))
for f in range(n):
    d.qpos[:]=Q[f];mujoco.mj_forward(m,d);r.update_scene(d,cam);s=r.scene
    # Transparent copy of the object at its goal pose.
    rot=gr[f].as_matrix();base=s.ngeom;count=0
    for i in range(base):
        g=s.geoms[i]
        if g.objid in og and g.objtype==mujoco.mjtObj.mjOBJ_GEOM and s.ngeom<s.maxgeom:
            k=g.objid;local_r=R.from_quat(m.geom_quat[k][[1,2,3,0]]).as_matrix();t=s.geoms[s.ngeom]
            mujoco.mjv_initGeom(t,g.type,g.size,gp[f]+rot@m.geom_pos[k],(rot@local_r).ravel(),np.array([.2,.8,1.,.35],np.float32));t.dataid=g.dataid;s.ngeom+=1
    img=r.render().copy();h,w=img.shape[:2]
    color=(255,80,60) if first>=0 and f>=first else (80,220,120)
    cv2.putText(img,a.title,(12,26),0,.62,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,f't={f/fps:.2f}s  frame {f}  error {100*ep[f]:.1f} cm / {np.degrees(er[f]):.0f} deg',(12,52),0,.55,color,2,cv2.LINE_AA)
    # Every object in the scene, so a failure outside this round's object is visible.
    row=136
    for k_,(e1_,e2_,fr_) in others.items():
        bad_=fr_>=0 and f>=fr_;cv2.putText(img,f'other obj{k_}: {100*e1_[f]:.1f} cm / {np.degrees(e2_[f]):.0f} deg'+(f'  FRONTIER {fr_/fps:.2f}s' if bad_ else ''),(12,row),0,.5,(255,80,60) if bad_ else (150,200,255),1,cv2.LINE_AA);row+=22
    if a.cause>=0 and abs(f-a.cause)<=6:cv2.putText(img,'CAUSE: fix targeted here',(12,80),0,.62,(255,210,60),2,cv2.LINE_AA)
    if first>=0 and abs(f-first)<=6:cv2.putText(img,'FRONTIER: lost or over budget',(12,106),0,.62,(255,80,60),2,cv2.LINE_AA)
    y=h-26;x=lambda v:int(14+(w-28)*v/max(1,n-1));cv2.line(img,(x(0),y),(x(n-1),y),(180,180,180),3)
    if first>=0:cv2.line(img,(x(first),y),(x(n-1),y),(255,80,60),5);cv2.putText(img,f'frontier {first/fps:.2f}s',(max(4,x(first)-30),y+20),0,.42,(255,80,60),1,cv2.LINE_AA)
    if a.cause>=0:cv2.circle(img,(x(a.cause),y),7,(255,210,60),-1);cv2.putText(img,f'cause {a.cause/fps:.2f}s',(max(4,x(a.cause)-30),y-12),0,.42,(255,210,60),1,cv2.LINE_AA)
    cv2.circle(img,(x(f),y),5,(255,255,255),-1);frames.append(img)
    if a.cause>=0 and f==a.cause:frames+=[img]*20
    if first>=0 and f==first:frames+=[img]*20
a.output.parent.mkdir(parents=True,exist_ok=True);imageio.mimsave(a.output,frames,fps=fps/1*1/a.slow*1,macro_block_size=None)
print(json.dumps(dict(video=str(a.output),frames=len(frames),first_break=first,cause=a.cause)))
