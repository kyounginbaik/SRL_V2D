import sys,json,re,numpy as np,mujoco
from pathlib import Path
e=Path(sys.argv[1]);run=Path(sys.argv[2]);m=mujoco.MjModel.from_binary_path(str(e/'yam_model.mjb'));d=mujoco.MjData(m);z=np.load(run/'physics.npz');Q=z['native_qpos'][:,0]
adr=lambda s:[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{s}_arm_joint{i}')] for i in range(1,7)]
def state(f):
    d.qpos[:]=Q[f];mujoco.mj_kinematics(m,d);out={}
    pm={s:d.xpos[m.body(f'{s}_arm_{s}_hand_C_MC').id].copy() for s in('left','right')}
    for s,o in(('left','right'),('right','left')):
        R=d.xmat[m.body(f'{s}_arm_{s}_hand_C_MC').id].reshape(3,3);n=R@np.array([.8,0,-.6]);i=pm[o]-pm[s];i/=np.linalg.norm(i);out[s]=dict(palm_dot_inward=round(float(n@i),2),palm_dot_up=round(float(n@np.array([0,0,1.])),2),arm=[round(float(Q[f][a]),2) for a in adr(s)])
    return out
print('frames',len(Q))
for f in(0,20,min(60,len(Q)-1),len(Q)-1):print(f,json.dumps(state(f)))
