"""Still of the YAM scene at the ready pose: from behind the arms and a close-up of each palm. No text."""
import os,sys,json,re
os.environ.setdefault('MUJOCO_GL','osmesa')
import numpy as np,mujoco,cv2
from pathlib import Path
e=Path(sys.argv[1]);out=sys.argv[2];m=mujoco.MjModel.from_binary_path(str(e/'yam_model.mjb'));m.vis.global_.offwidth=1920;m.vis.global_.offheight=1080;d=mujoco.MjData(m);READY=[0.,1.536,.945,-.61,0.,0.]
for j in range(m.njnt):
    k=re.search(r'_arm_joint([1-6])$',m.joint(j).name)
    if k:d.qpos[m.jnt_qposadr[j]]=READY[int(k.group(1))-1]
mujoco.mj_forward(m,d);p=json.loads((e/'yam_placement.json').read_text())
for g in range(m.ngeom):
    n=m.body(int(m.geom_bodyid[g])).name
    if n.startswith(('left_','right_')) and (m.geom_contype[g] or m.geom_conaffinity[g]):m.geom_group[g]=5
    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:m.geom_group[g]=5
    if n=='support':m.geom_rgba[g]=(.78,.74,.68,1)
r=mujoco.Renderer(m,1080,1920);pl={s:d.xpos[m.body(f'{s}_arm_{s}_hand_C_MC').id].copy() for s in('left','right')};mid=(pl['left']+pl['right'])/2
def shot(name,look,az,el,dist):
    c=mujoco.MjvCamera();c.lookat[:]=look;c.azimuth=az;c.elevation=el;c.distance=dist;r.update_scene(d,c);cv2.imwrite(f'{out}_{name}.png',r.render()[...,::-1])
shot('behind',mid+np.array([0,.0,-.1]),90,-28,1.5);shot('front',mid+np.array([0,0,-.1]),-90,-22,1.5);shot('right_hand_palmside',pl['right'],180,-5,.5);shot('left_hand_palmside',pl['left'],0,-5,.5)
print(json.dumps({s:pl[s].round(3).tolist() for s in pl}))
