"""Frame-by-frame: what breaks a placed grasp under the real object goal?"""
import sys,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab,FPS,SUBSTEPS
root=Path(__file__).parent.parent/'runs';fit=root/'surface_anchor_fit_v7'
lab=Lab(fit/'native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',19);m,d=lab.m,lab.d
z=np.load(fit/'candidate_018.npz');star=z['qpos']
right={int(m.jnt_bodyid[j]) for j in lab.joints};log=[]
orig=mujoco.mj_step
def hand_force():
    total=np.zeros(3);n=0;names=set();f6=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i];a,b=m.geom_bodyid[c.geom]
        if lab.obj in (a,b) and (a in right or b in right):
            mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3];total+=(fw if b==lab.obj else -fw);n+=1;names.add(m.body(int(a if b==lab.obj else b)).name[23:])
    return total,n,sorted(names)
o0=star[lab.qo:lab.qo+3].copy();r0=R.from_quat(star[lab.qo+3:lab.qo+7][[1,2,3,0]])
print('goal at onset vs rest: dpos cm',round(100*float(np.linalg.norm(lab.gp[lab.onset]-o0)),2),'drot deg',round(float(np.degrees((lab.gr[lab.onset]*r0.inv()).magnitude())),1),'onset',lab.onset)
class Probe:
    pass
# Replay Lab.run but sample state each control frame via a wrapped mj_step counter.
count=[0];rows=[]
def wrapped(mm,dd):
    orig(mm,dd);count[0]+=1
    if count[0]%SUBSTEPS==0:
        t,n,names=hand_force();rows.append((count[0]//SUBSTEPS,d.qpos[lab.qo:lab.qo+3].copy(),R.from_quat(d.qpos[lab.qo+3:lab.qo+7][[1,2,3,0]]),d.qpos[29:32].copy(),R.from_matrix(d.xmat[lab.body].reshape(3,3).copy()),float(t@lab.up),n,names))
mujoco.mj_step=wrapped
r=lab.run(star,z['patch_positions'],z['patch_normals'],z['hand_geom_ids'],8.,horizon=30);mujoco.mj_step=orig;print(r)
pre=int(.5*FPS);o_i=None
for row in rows:
    f,o,ro,w,rw,up,n,names=row
    if f==pre:o_i=(o.copy(),ro,w.copy(),rw)
    if f<pre-3:continue
    if o_i is None:print(f,'prelift object lift mm',round(1e3*float((o-o0)@lab.up),1),'upF/mg',round(up/lab.mg,2),n,names);continue
    rel=(o_i[3].inv()*rw)*(o_i[1].inv()*ro).inv();k=min(lab.onset+f-pre,len(lab.gp_raw)-1)
    print(f,'gz',round(float(lab.gp_raw[k][2]),3),'turn',round(float(np.degrees((lab.gr_raw[k]*r0.inv()).magnitude())),1),'wristMove cm',round(100*float(np.linalg.norm(w-o_i[2])),1),'objMove cm',round(100*float(np.linalg.norm(o-o_i[0])),1),'slipDeg',round(float(np.degrees(rel.magnitude())),1),'upF/mg',round(up/lab.mg,2),n,names)
