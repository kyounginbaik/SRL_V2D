"""Render a run from the recording's own moving RGB camera, and put it beside the source video when one is given."""
import argparse,json,os,subprocess
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,imageio
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as RR
p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--title',default='');p.add_argument('--world',type=int,default=0);p.add_argument('--object',type=int,default=0);a=p.parse_args()
meta=json.loads((a.episode/'recon_meta.json').read_text());pad=int(meta.get('pad',0));names=[o['name'] for o in meta['objects']]
z=np.load(a.run/'physics.npz');cap=int(z['native_capture_world_ids'][a.world]);Q=z['native_qpos'][:,a.world];poses=z['object_pose'][:,cap];fps=float(z['fps'])
b=np.load(a.bundle/'object_poses.npz');K=json.loads((a.bundle/'intrinsics.json').read_text());W=b['world_from_rgb'];O=b['world_from_object'];bn=[str(x) for x in b['names']]
k=max(meta['movers'],key=lambda x:x['move_cm'])['index'];Ow=O[:,bn.index(names[k])]
def mat(pq):
    T=np.eye(4);T[:3,3]=pq[:3];T[:3,:3]=RR.from_quat(pq[3:7]).as_matrix();return T
# Simulator world from recording world, from the goal path of the object; pad frames were added in front.
fits={}
for off in sorted({pad,0}):
    pts=[(poses[f+off,min(k,poses.shape[1]-1),7:10],Ow[f,:3,3]) for f in range(0,min(len(Ow),len(poses)-off),3)];A,B=np.array([x[1] for x in pts]),np.array([x[0] for x in pts]);ca,cb=A.mean(0),B.mean(0);Rf,_=RR.align_vectors(B-cb,A-ca);res=np.linalg.norm(Rf.apply(A-ca)+cb-B,axis=1);fits[off]=(float(np.median(res)),Rf,ca,cb)
off=min(fits,key=lambda x:fits[x][0]);res,Rf,ca,cb=fits[off];S=np.eye(4);S[:3,:3]=Rf.as_matrix();S[:3,3]=cb-Rf.apply(ca);up=S[:3,:3]@np.array([0,0,1.])
print(json.dumps(dict(pad=pad,frame_offset=off,fit_cm=round(100*res,2),fits={str(o):round(100*v[0],2) for o,v in fits.items()})),flush=True)
m=mujoco.MjModel.from_binary_path(str(next(p for p in (a.episode/'yam/yam_model_visual.mjb',a.episode/'yam/yam_model.mjb') if p.is_file())))
VENDOR=m.geom_rgba.copy();m.geom_matid[:]=-1
free=[j for j in range(m.njnt) if m.jnt_type[j]==0];bod=lambda i:int(m.jnt_bodyid[min(free,key=lambda j:float(np.linalg.norm(Q[0][m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-poses[0,i,:3])))])
PAL={0:(.10,.70,.75,1),1:(.16,.40,.90,1),2:(.30,.75,.40,1)}
rob=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_'))]
for g in range(m.ngeom):
    n=m.body(int(m.geom_bodyid[g])).name
    if n=='support':m.geom_rgba[g]=(.66,.38,.18,1)
for i in range(poses.shape[1]):
    for g in np.flatnonzero(m.geom_bodyid==bod(i)):m.geom_rgba[g]=PAL[i%3]
if any(m.geom_contype[g]==0 and m.geom_conaffinity[g]==0 for g in rob):
    for g in rob:
        if m.geom_contype[g]!=0 or m.geom_conaffinity[g]!=0:m.geom_group[g]=5
        else:m.geom_group[g]=1;m.geom_rgba[g]=VENDOR[g]
m.geom_group[m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE]=5
# The robot's base stands near the wearer's head, so its arms would hide the workspace from this camera.
for g in rob:
    if '_arm_' in m.body(int(m.geom_bodyid[g])).name:m.geom_rgba[g][3]=.22
w,h=int(K['width']),int(K['height']);m.vis.global_.fovy=float(np.degrees(2*np.arctan(.5*h/K['fy'])));m.vis.global_.offwidth=w;m.vis.global_.offheight=h;m.vis.quality.offsamples=8;m.vis.map.znear=.01;m.vis.headlight.ambient[:]=.5;m.vis.headlight.diffuse[:]=.45;m.vis.headlight.specular[:]=.1
import booth;BOOTH=booth.boxes(a.episode/'yam');d=mujoco.MjData(m);r=mujoco.Renderer(m,h,w);cam=mujoco.MjvCamera()
a.output.parent.mkdir(parents=True,exist_ok=True);sim=a.output.with_name(a.output.stem+'_sim.mp4');wr=imageio.get_writer(sim,fps=fps,codec='libx264',ffmpeg_params=['-crf','18','-pix_fmt','yuv420p'],macro_block_size=None)
for f in range(len(Q)):
    i=min(max(f-off,0),len(W)-1);T=S@W[i];d.qpos[:]=Q[f];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d);r.update_scene(d,cam);s=r.scene;booth.add(s,BOOTH)
    # OpenCV camera axes: x right, y down, z forward.
    for c in s.camera:c.pos[:]=T[:3,3];c.forward[:]=T[:3,2];c.up[:]=-T[:3,1]
    im=r.render();img=Image.fromarray(im);dr=ImageDraw.Draw(img);dr.text((18,14),f"{a.title}  frame {f}",fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0),font_size=26);wr.append_data(np.asarray(img))
wr.close()
if a.source and a.source.is_file():
    # Source frame i is simulator frame i+off; the source holds still through the settling frames added in front and after its end.
    import imageio_ffmpeg
    dur=len(Q)/fps;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(a.source),'-i',str(sim),'-filter_complex',f"[0:v]scale={w}:{h},setsar=1,tpad=start_duration={off/fps}:start_mode=clone:stop_duration=5:stop_mode=clone[a];[1:v]setsar=1[b];[a][b]hstack=inputs=2[v]",'-map','[v]','-t',str(dur),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(a.output)],check=True)
print('RENDERED',a.output if a.source else sim,flush=True)
