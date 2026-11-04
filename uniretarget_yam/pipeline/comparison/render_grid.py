import os,sys,json
os.environ["MUJOCO_GL"]="egl"
import numpy as np,mujoco,imageio
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,"/uniretarget_yam/code")
import yam_loop as Y
Y.ONSET_FLOOR=1.4
t=Y.YamTask("/uniretarget_yam/episodes/episode_000037","right",0);m=t.m;d=mujoco.MjData(m)
W,H=640,480;m.vis.global_.offwidth=W;m.vis.global_.offheight=H;r=mujoco.Renderer(m,H,W)
def qmap(names):
    out=[]
    for n in names:
        n=str(n)
        if n.startswith(("R_arm_j","L_arm_j")):
            j=("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]
        else:
            j=("right" if n.startswith("right") else "left")+"_arm_"+n
        out.append(int(m.jnt_qposadr[m.joint(j).id]))
    return out
CLIPS=json.loads(sys.argv[1]);out=sys.argv[2]
data=[(lab,np.load(f"/vid/{n}.npz")) for n,lab in CLIPS]
cam=mujoco.MjvCamera();cam.lookat[:]=t.o0+t.up*0.17;cam.distance=0.95;cam.azimuth=125;cam.elevation=-18
try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",20)
except Exception: font=ImageFont.load_default()
F=min(len(z["q"]) for _,z in data);frames=[];stills=[]
maps=[qmap(z["names"]) for _,z in data]
for f in range(F):
    tiles=[]
    for (lab,z),mp in zip(data,maps):
        d.qpos[:]=t.base_q;d.qpos[mp]=z["q"][f];x=z["obj"][f];d.qpos[t.qo:t.qo+3]=x[:3];d.qpos[t.qo+3:t.qo+7]=x[[6,3,4,5]];mujoco.mj_kinematics(m,d)
        r.update_scene(d,cam);g=r.scene
        if g.ngeom<g.maxgeom:
            mujoco.mjv_initGeom(g.geoms[g.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012,0,0]),z["goal"][f].astype(float),np.eye(3).flatten(),np.array([.1,.9,.2,.8],dtype=np.float32));g.ngeom+=1
        im=Image.fromarray(r.render().copy());dr=ImageDraw.Draw(im);rise=(z["obj"][:f+1,2].max()-z["obj"][0,2])*100
        dr.rectangle([0,0,W,30],fill=(0,0,0));dr.text((8,4),lab,fill=(255,255,255),font=font);dr.text((8,H-28),f"t={f/20:4.1f}s  max rise {rise:4.1f} cm",fill=(255,255,255),font=font)
        tiles.append(np.asarray(im))
    n=len(tiles);c=2;rows=[np.hstack(tiles[i:i+c]) for i in range(0,n,c)]
    fr=np.vstack(rows);frames.append(fr)
    if f in (20,42,50,70):Image.fromarray(fr).save(out.replace(".mp4",f"_f{f}.png"))
imageio.mimsave(out,frames,fps=20,codec="libx264",quality=7,macro_block_size=1)
print("wrote",out,F,flush=True)
