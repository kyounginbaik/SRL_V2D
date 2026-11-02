"""Presentation-quality trace video: supersampled render, lit materials, clean background, real type."""
import argparse,json,os
os.environ.setdefault('MUJOCO_GL','osmesa')
from pathlib import Path
import numpy as np,mujoco,cv2,imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation as R
W,H=1920,1080;SS=2;FONT_DIRS=('/usr/share/fonts/truetype/dejavu','/opt/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf')
def font(size,bold=False):
    # Google Sans (SIL OFL, code/fonts): a variable font, so weight is set through its axes.
    f=ImageFont.truetype(str(Path(__file__).parent/'fonts/GoogleSans.ttf'),size);axes=[]
    for ax in f.get_variation_axes():
        name=ax['name'];name=name.decode() if isinstance(name,bytes) else name;axes.append((700 if bold else 400) if 'eight' in name else (min(max(size,ax['minimum']),ax['maximum']) if 'ptical' in name else ax['default']))
    f.set_variation_by_axes(axes);return f
def loop_frontier(ep,er,total):
    n=np.arange(1,len(ep)+1);limit=.7*n+.3*total;over=np.flatnonzero((ep>.15)|(er>np.radians(150.))|(np.cumsum(ep)>.03*limit)|(np.cumsum(er)>np.radians(30.)*limit));return int(over[0]) if len(over) else -1
def errors(o):
    i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);return gp,gr,np.linalg.norm(o[:,:3]-gp,axis=1),(R.from_quat(o[:,3:7]).inv()*gr).magnitude()
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=0);p.add_argument('--object',type=int,default=0);p.add_argument('--title',default='');p.add_argument('--subtitle',default='');p.add_argument('--cause',type=int,default=-1);p.add_argument('--output',type=Path,required=True);p.add_argument('--still',type=int);p.add_argument('--slow',type=int,default=2);p.add_argument('--ghost',type=Path);p.add_argument('--ghost_side',default='right_');p.add_argument('--edits',default='');p.add_argument('--kinematic',action='store_true');a=p.parse_args()
m=mujoco.MjModel.from_binary_path(str(a.model));z=np.load(a.run/'physics.npz');cap=int(z['native_capture_world_ids'][a.world]);Q=z['native_qpos'][:,a.world];poses=z['object_pose'][:,cap];fps=float(z['fps'])
free=[j for j in range(m.njnt) if m.jnt_type[j]==0];body=lambda k:int(m.jnt_bodyid[min(free,key=lambda j:float(np.linalg.norm(z['native_qpos'][0,0][m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]-poses[0,k,:3])))]);obj=body(a.object)
# Materials: warm table, graphite hands, saturated object colours, soft studio lights.
PALETTE={0:(.86,.42,.20,1),1:(.16,.52,.86,1),2:(.30,.70,.40,1)}
VENDOR=m.geom_rgba.copy()
m.geom_matid[:]=-1
for g in range(m.ngeom):
    n=m.body(int(m.geom_bodyid[g])).name
    if n.startswith(('left_','right_')):m.geom_rgba[g]=(.20,.22,.26,1) if n.startswith('right_') else (.33,.35,.40,1)
    elif n=='support':m.geom_rgba[g]=(.78,.74,.68,1)
    elif m.geom(g).name.startswith('yam_'):m.geom_rgba[g]=VENDOR[g]
    elif int(m.geom_bodyid[g])==0:m.geom_rgba[g]=(.93,.93,.94,1)
for k in range(poses.shape[1]):
    for g in np.flatnonzero(m.geom_bodyid==body(k)):m.geom_rgba[g]=PALETTE[k%3]
# With vendor visual meshes present, show those and hide the robot's collision shapes.
robot_geoms=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_'))]
if any(m.geom_contype[g]==0 and m.geom_conaffinity[g]==0 for g in robot_geoms):
    for g in robot_geoms:
        if m.geom_contype[g]!=0 or m.geom_conaffinity[g]!=0:m.geom_group[g]=5
        else:
            m.geom_group[g]=1;m.geom_rgba[g]=VENDOR[g]  # The URDF's own materials: black YAM, Sharpa shell, green pads.
m.geom_group[m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE]=5  # Default render options hide group 5: no floor.
m.vis.quality.numslices=128;m.vis.quality.numstacks=64;m.vis.global_.offwidth=W*SS;m.vis.global_.offheight=H*SS;m.vis.quality.shadowsize=8192;m.vis.quality.offsamples=8;m.vis.headlight.ambient[:]=.42;m.vis.headlight.diffuse[:]=.42;m.vis.headlight.specular[:]=.12;m.vis.map.znear=.01
d=mujoco.MjData(m);r=mujoco.Renderer(m,H*SS,W*SS);seg=mujoco.Renderer(m,H*SS,W*SS);seg.enable_segmentation_rendering()
gp,gr,ep,er=errors(poses[:,a.object]);first=loop_frontier(ep,er,len(ep)+1);n=len(Q);others={k:errors(poses[:,k]) for k in range(poses.shape[1]) if k!=a.object};others={k:(v[2],v[3],loop_frontier(v[2],v[3],len(v[2])+1)) for k,v in others.items()}
# Camera frames everything that moves, from a fixed three-quarter view.
# Frames every object and both wrists over the whole round, with room for the panels.
wrists=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==1]+[b for b in range(m.nbody) if m.body(b).name.endswith('_hand_C_MC')];track=[]
for f in range(0,n,max(1,n//40)):
    d.qpos[:]=Q[f];mujoco.mj_kinematics(m,d);track+= [d.xpos[b].copy() for b in wrists]+[poses[f,k,:3] for k in range(poses.shape[1])]
track=np.array(track);lo,hi=np.percentile(track,3,axis=0),np.percentile(track,97,axis=0);cam=mujoco.MjvCamera();cam.lookat[:]=.5*(lo+hi)+np.array([0,0,.02]);cam.distance=1.55*max(.42,float(np.linalg.norm(hi-lo)));cam.elevation=-20;cam.azimuth=118
if any(m.body(b).name.endswith('_arm_base_link') for b in range(m.nbody)):
    # The arms face the table along recon +y, so look back at them from the far side and above.
    cam.azimuth=-68;cam.elevation=-34;cam.distance=max(cam.distance,1.25);cam.lookat[:]=np.median(poses[:n,:,:3].reshape(-1,3),axis=0)+np.array([0,-.08,.06])
vy=np.linspace(0,1,H*SS)[:,None,None];top=np.array([232,236,241.]);bot=np.array([186,192,200.]);bg=(top*(1-vy)+bot*vy)*np.ones((1,W*SS,1))
floor=[g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE];F_T=font(34,True);F_S=font(26);F_B=font(30,True);F_M=font(26);F_L=font(22);INK=(28,32,38);RED=(205,48,42);GRN=(30,140,84);AMB=(214,140,20);MUT=(92,98,108);BLUE=(36,104,190)
GHOST_ID=100000;ROBOT_GHOST_ID=100001
ghost=np.load(a.ghost) if a.ghost else None;dg=mujoco.MjData(m);edits=[int(x) for x in a.edits.split(',') if x]
def add_robot_ghost(s,f):
    # The original reference as a second robot, only on the arm that differs and only where it differs.
    dg.qpos[:]=ghost['qpos'][f];mujoco.mj_kinematics(m,dg);base=s.ngeom
    for i in range(base):
        g=s.geoms[i]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid<m.ngeom and s.ngeom<s.maxgeom and m.body(int(m.geom_bodyid[g.objid])).name.startswith(a.ghost_side) and not m.body(int(m.geom_bodyid[g.objid])).name.endswith('base_link'):
            k=g.objid;t=s.geoms[s.ngeom];mujoco.mjv_initGeom(t,g.type,g.size,dg.geom_xpos[k],dg.geom_xmat[k],np.array([.35,.68,1.,1.],np.float32));t.dataid=g.dataid;t.objtype=mujoco.mjtObj.mjOBJ_GEOM;t.objid=ROBOT_GHOST_ID;s.ngeom+=1
def add_ghost(s,f):
    rot=gr[f].as_matrix();base=s.ngeom
    for i in range(base):
        g=s.geoms[i]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid<m.ngeom and m.geom_bodyid[g.objid]==obj and s.ngeom<s.maxgeom:
            k=g.objid;lr=R.from_quat(m.geom_quat[k][[1,2,3,0]]).as_matrix();t=s.geoms[s.ngeom];mujoco.mjv_initGeom(t,g.type,g.size,gp[f]+rot@m.geom_pos[k],(rot@lr).ravel(),np.array([.10,.80,.48,1.],np.float32));t.dataid=g.dataid;t.objtype=mujoco.mjtObj.mjOBJ_GEOM;t.objid=GHOST_ID;s.ngeom+=1
def frame(f):
    d.qpos[:]=Q[f];mujoco.mj_forward(m,d);r.update_scene(d,cam);s=r.scene;s.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=1;s.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
    if s.nlight<8:
        l=s.lights[s.nlight];l.type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL;l.pos[:]=cam.lookat+np.array([.6,-.8,1.6]);l.dir[:]=(cam.lookat-l.pos)/np.linalg.norm(cam.lookat-l.pos);l.diffuse[:]=.55;l.ambient[:]=.05;l.specular[:]=.15;l.castshadow=1;s.nlight+=1
    rgb=r.render().astype(float);seg.update_scene(d,cam);ids=seg.render()[...,0];mask=(ids<0)|np.isin(ids,floor);rgb[mask]=bg[mask]
    # Goal ghost: rendered in its own pass so depth stays right, then blended over the composite.
    if not a.kinematic:add_ghost(r.scene,f);lit=r.render().astype(float);seg.update_scene(d,cam);add_ghost(seg.scene,f);over=seg.render()[...,0]==GHOST_ID;rgb[over]=.55*rgb[over]+.45*lit[over]
    if ghost is not None and (ghost['palm_cm'][f]>.2 or ghost['palm_deg'][f]>2):
        add_robot_ghost(r.scene,f);lit=r.render().astype(float);add_robot_ghost(seg.scene,f);over=seg.render()[...,0]==ROBOT_GHOST_ID;rgb[over]=.45*rgb[over]+.55*lit[over]
    im=cv2.resize(rgb.astype(np.uint8),(W,H),interpolation=cv2.INTER_AREA);img=Image.fromarray(im).convert('RGBA');ov=Image.new('RGBA',img.size,(0,0,0,0));dr=ImageDraw.Draw(ov)
    # One panel, sized to its contents: title, time, current error, episode mean (what the official metric judges), limit.
    bad=first>=0 and f>=first;f1=f+1;mc,md=100*ep[:f1].mean(),np.degrees(er[:f1].mean());VX=148
    other_text=[(f'other object {k}:  {100*e1[f]:.1f} cm / {np.degrees(e2[f]):.0f}°'+('   past its frontier' if fr>=0 and f>=fr else ''),RED if fr>=0 and f>=fr else BLUE) for k,(e1,e2,fr) in others.items()]
    right=60+max(dr.textlength(a.title,font=F_T),dr.textlength(a.subtitle,font=F_S) if a.subtitle else 0,VX-60+dr.textlength('00.00 cm  /  000.0°',font=F_B),dr.textlength('t = 00.00 s   frame 000',font=F_M),*(dr.textlength(x,font=F_L) for x,_ in other_text))+24
    y=102+(38 if a.subtitle else 0);dr.rounded_rectangle((36,32,right,y+36*((1 if a.kinematic else 4)+(ghost is not None))+34*(0 if a.kinematic else len(other_text))+14),18,fill=(255,255,255,222));dr.text((60,50),a.title,font=F_T,fill=INK)
    if a.subtitle:dr.text((60,102),a.subtitle,font=F_S,fill=MUT)
    dr.text((60,y),f't = {f/fps:.2f} s   frame {f}',font=F_M,fill=MUT);y+=36
    if not a.kinematic:
        dr.text((60,y),'now',font=F_M,fill=MUT);dr.text((VX,y-2),f'{100*ep[f]:.1f} cm  /  {np.degrees(er[f]):.0f}°',font=F_B,fill=RED if bad else (AMB if ep[f]>=.03 or er[f]>=np.radians(30) else GRN));y+=36
        dr.text((60,y),'mean',font=F_M,fill=MUT);dr.text((VX,y-2),f'{mc:.2f} cm  /  {md:.1f}°',font=F_B,fill=RED if mc>=3 or md>=30 else GRN);y+=36
        dr.text((60,y),'limit',font=F_M,fill=MUT);dr.text((VX,y),'3 cm  /  30°',font=F_M,fill=MUT);y+=36
        for x,c in other_text:dr.text((60,y),x,font=F_L,fill=c);y+=34
    if ghost is not None:dr.text((60,y),'edit',font=F_M,fill=MUT);dr.text((VX,y-2),f"{ghost['palm_cm'][f]:.1f} cm  /  {ghost['palm_deg'][f]:.0f}°",font=F_B,fill=BLUE);y+=36
    if a.cause>=0 and abs(f-a.cause)<=6:dr.rounded_rectangle((36,H-214,560,H-150),14,fill=AMB+(235,));dr.text((58,H-200),'Traced cause: fix targets here',font=F_M,fill=(255,255,255))
    if first>=0 and abs(f-first)<=6:dr.rounded_rectangle((590,H-214,1240,H-150),14,fill=RED+(235,));dr.text((612,H-200),'Frontier: lost, or error budget used up',font=F_M,fill=(255,255,255))
    x0,x1,yb=60,W-60,H-84;xf=lambda v:int(x0+(x1-x0)*v/max(1,n-1));dr.rounded_rectangle((36,H-132,W-36,H-30),18,fill=(255,255,255,222));dr.rounded_rectangle((x0,yb-5,x1,yb+5),5,fill=(205,210,216,255));dr.rounded_rectangle((x0,yb-5,xf(f),yb+5),5,fill=(120,128,140,255))
    if first>=0:dr.rounded_rectangle((xf(first),yb-5,x1,yb+5),5,fill=RED+(255,));dr.text((min(xf(first),x1-210),yb+12),f'frontier {first/fps:.2f} s',font=F_L,fill=RED)
    elif not a.kinematic:dr.text((x1-230,yb+12),'no frontier: full episode',font=F_L,fill=GRN)
    if a.cause>=0:dr.ellipse((xf(a.cause)-11,yb-11,xf(a.cause)+11,yb+11),fill=AMB+(255,));dr.text((min(x1-130,max(x0,xf(a.cause)-70)),yb-42),f'cause {a.cause/fps:.2f} s',font=F_L,fill=AMB)
    dr.ellipse((xf(f)-9,yb-9,xf(f)+9,yb+9),fill=INK+(255,));dr.text((x0,yb+12),'solid = refined kinematics      blue = original kinematics      object = reference path'+('      blue dots = edits' if edits else '') if a.kinematic else 'green = goal pose'+('      solid = refined reference      blue = original reference      blue dots = edits' if ghost is not None else ''),font=F_L,fill=MUT)
    for e in edits:dr.ellipse((xf(e)-8,yb-8,xf(e)+8,yb+8),fill=BLUE+(255,));dr.text((max(x0,xf(e)-28),yb-40),f'{e/fps:.2f} s',font=F_L,fill=BLUE)
    return np.asarray(Image.alpha_composite(img,ov).convert('RGB'))
if a.still is not None:
    import time;t=time.monotonic();im=frame(a.still);print('seconds per frame',round(time.monotonic()-t,2));Image.fromarray(im).save(a.output)
else:
    a.output.parent.mkdir(parents=True,exist_ok=True);w=imageio.get_writer(a.output,fps=fps/a.slow*2,codec='libx264',quality=None,ffmpeg_params=['-crf','17','-preset','slow','-pix_fmt','yuv420p'],macro_block_size=None)
    for f in range(n):
        im=frame(f);rep=2
        if f==a.cause or f==first:rep=2+int(1.5*fps/a.slow*2)
        for _ in range(rep):w.append_data(im)
    w.close();print(json.dumps(dict(video=str(a.output),first=first,cause=a.cause)))
