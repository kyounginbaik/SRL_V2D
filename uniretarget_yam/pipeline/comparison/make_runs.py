"""Turn recorded Newton clips into the physics.npz layout render_hq.py reads."""
import sys,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,"/uniretarget_yam/code")
import yam_loop as Y
Y.ONSET_FLOOR=1.4
import os
t=Y.YamTask(os.environ.get("RECON_EP","/uniretarget_yam/episodes/episode_000037"),os.environ.get("RECON_SIDE","right"),int(os.environ.get("RECON_OBJ","0")));m=t.m;d=mujoco.MjData(m)
def qmap(names):
    out=[]
    for n in map(str,names):
        j=(("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]) if n.startswith(("R_arm_j","L_arm_j")) else (("right" if n.startswith("right") else "left")+"_arm_"+n)
        out.append(int(m.jnt_qposadr[m.joint(j).id]))
    return out
def save(out,Q,P,ghost=None):
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"physics.npz",native_qpos=Q[:,None],object_pose=P[:,None,None],native_capture_world_ids=np.array([0]),fps=np.array(20.))
    if ghost is not None:np.savez_compressed(out/"ghost.npz",**ghost)
for name in sys.argv[1].split(","):
    z=np.load(f"/vid/{name}.npz");F=len(z["q"]);mp=qmap(z["names"]);Q=np.tile(t.base_q,(F,1));Q[:,mp]=z["q"];x=z["obj"];Q[:,t.qo:t.qo+3]=x[:,:3];Q[:,t.qo+3:t.qo+7]=x[:,[6,3,4,5]]
    # Every object moves as it did in physics, not only the one this run is judged on.
    if "objs" in z:
        for k,q_ in enumerate(t.object_q):Q[:,q_:q_+3]=z["objs"][:,k,:3];Q[:,q_+3:q_+7]=z["objs"][:,k][:,[6,3,4,5]]
        print(name,"objects moved cm",[round(100*float(np.linalg.norm(z["objs"][:,k,:3]-z["objs"][0,k,:3],axis=1).max()),1) for k in range(z["objs"].shape[1])],flush=True)
    P=np.concatenate([x,z["goal"],z["goal_quat"]],axis=1);save(Path(f"/vid/run_{name}"),Q,P)
    e=np.linalg.norm(x[:,:3]-(R.from_quat(x[0,3:7])*R.from_quat(z["goal_quat"][0]).inv()).apply(z["goal"]-z["goal"][0])-x[0,:3],axis=1)
    print(name,"mean_cm",round(100*float(e.mean()),2),"rise",round(100*float(x[:,2].max()-x[0,2]),1),flush=True)
# Reference refinement: both robots kinematic, the object on its aligned goal path.
z=np.load("/vid/%s.npz"%sys.argv[2]);F=len(z["q"]);mp=qmap(z["names"]);x=z["obj"];g,gq=z["goal"],z["goal_quat"];al=R.from_quat(x[0,3:7])*R.from_quat(gq[0]).inv();gp=al.apply(g)+x[0,:3]-al.apply(g[0]);ga=(al*R.from_quat(gq)).as_quat()
def kin(q):
    Q=np.tile(t.base_q,(F,1));Q[:,mp]=q[:F];Q[:,t.qo:t.qo+3]=gp;Q[:,t.qo+3:t.qo+7]=ga[:,[3,0,1,2]]
    if "objs" in z:
        for k,q_ in enumerate(t.object_q):
            if q_==t.qo:continue
            o0=z["objs"][0,k];al_=R.from_quat(o0[3:7])*R.from_quat(z["goal_quats"][0,k]).inv();Q[:,q_:q_+3]=al_.apply(z["goals"][:,k]-z["goals"][0,k])+o0[:3];Q[:,q_+3:q_+7]=(al_*R.from_quat(z["goal_quats"][:,k])).as_quat()[:,[3,0,1,2]]
    return Q
Qr,Qo=kin(z["ref_q"]),kin(z["human_q"]);dp=[];dr=[]
for f in range(F):
    d.qpos[:]=Qr[f];mujoco.mj_kinematics(m,d);p1,r1=d.xpos[t.palm].copy(),R.from_matrix(d.xmat[t.palm].reshape(3,3).copy());d.qpos[:]=Qo[f];mujoco.mj_kinematics(m,d);dp.append(np.linalg.norm(p1-d.xpos[t.palm]));dr.append((r1*R.from_matrix(d.xmat[t.palm].reshape(3,3).copy()).inv()).magnitude())
dp,dr=np.array(dp),np.array(dr);P=np.concatenate([gp,ga,g,gq],axis=1);save(Path("/vid/run_refinement"),Qr,P,dict(qpos=Qo,palm_cm=100*dp,palm_deg=np.degrees(dr)))
print("refinement palm cm mean",round(100*float(dp.mean()),1),"max",round(100*float(dp.max()),1),"deg max",round(float(np.degrees(dr.max())),0),flush=True)
