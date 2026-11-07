import sys, numpy as np, mujoco
import planner as PL, wmetric as W
from scipy.spatial.transform import Rotation as R
f,sq=200,.2
m,d,b=PL.setup(); RF=W.RF; k=RF['q']; dt=m.opt.timestep; sub=int(round(.02/dt))
j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'right_object_joint'); pa=m.jnt_qposadr[j]; da=m.jnt_dofadr[j]
print('ref FK sanity: does base_from reproduce frame 300?')
Xt=PL.wrist_T(m,d,b,k[300]); q=PL.base_from(m,d,b,Xt,k[290]); print(' base err',np.abs(q-k[300,:6]).max(), 'pose err',np.abs(PL.wrist_T(m,d,b,np.r_[q,k[290,6:]])[:3,3]-Xt[:3,3]).max())
X0=np.linalg.inv(PL.wrist_T(m,d,b,k[f]))@PL.pen_T(k[f,-7:]); qf=k[f,6:28].copy(); lo,hi=m.actuator_ctrlrange[6:,0],m.actuator_ctrlrange[6:,1]
goal=np.clip(qf+sq*np.array([any(s in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for s in('FE','PIP','DIP','IP')) for i in range(6,28)],float),lo,hi); opn=np.clip(qf*.45,lo,hi)
pen0=RF['pen'][0].copy(); base=PL.base_from(m,d,b,PL.pen_T(pen0)@np.linalg.inv(X0),k[0]); mujoco.mj_resetData(m,d); d.qpos[:6]=base; d.qpos[6:28]=opn; d.qpos[pa:pa+7]=pen0; d.ctrl[:6]=base; d.ctrl[6:]=opn; mujoco.mj_forward(m,d)
def pair(): 
    hp=sum(1 for i in range(d.ncon) if ((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,d.contact[i].geom1) or '').startswith('right_object'))!=((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,d.contact[i].geom2) or '').startswith('right_object')) )
    return hp
print('start contacts',d.ncon,'hand-pen',pair())
for i in range(int(.8/dt)):
    d.ctrl[6:]=opn+(goal-opn)*min(1,i/(.6/dt)); d.qpos[pa:pa+7]=pen0; d.qvel[da:da+6]=0; mujoco.mj_step(m,d)
print('closed contacts',d.ncon,'hand-pen',pair(),'finger q err',np.abs(d.qpos[6:28]-goal).max())
for i in range(int(.7/dt)): mujoco.mj_step(m,d)
p=d.qpos[pa:pa+7].copy(); print('after hold pen move cm',np.linalg.norm(p[:3]-pen0[:3])*100,'contacts',pair(),'wrist-base err',np.abs(d.qpos[:6]-base).max())
Xh=np.linalg.inv(PL.wrist_T(m,d,b,d.qpos.copy()))@PL.pen_T(p); mujoco.mj_forward(m,d); md=mujoco.MjData(m)
for t in range(RF['n']):
    Xp=PL.pen_T(RF['pen'][t]); Xw=Xp@np.linalg.inv(Xh); md.qpos[:]=d.qpos; base=PL.base_from(m,md,b,Xw,d.qpos.copy()); d.ctrl[:6]=base
    for _ in range(sub): mujoco.mj_step(m,d)
    if t%40==0:
        cur=PL.pen_T(d.qpos[pa:pa+7]); Xnow=PL.wrist_T(m,md,b,d.qpos.copy()); inhand=np.linalg.inv(Xnow)@cur; 
        print(t,'pen err',round(np.linalg.norm(cur[:3,3]-Xp[:3,3])*100,1),'wrist err',round(np.linalg.norm(Xnow[:3,3]-Xw[:3,3])*100,2),'wrist rot',round(R.from_matrix(Xnow[:3,:3]@Xw[:3,:3].T).magnitude(),3),'in-hand slip cm',round(np.linalg.norm(inhand[:3,3]-Xh[:3,3])*100,1),'rad',round(R.from_matrix(inhand[:3,:3]@Xh[:3,:3].T).magnitude(),2),'hand-pen',pair(),'gap',)
