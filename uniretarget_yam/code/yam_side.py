"""Which hand moves each object in the demonstration, and when: decides the YAM side and search order per episode."""
import sys,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
for e in sys.argv[1:]:
    e=Path(e)
    try:
        n=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));J=np.load(e/'plan.npz')['joint_q'][0];z=np.load(e/'baseline/physics.npz');P=z['object_pose'][:,0];d=mujoco.MjData(n);nq=n.nq-7*P.shape[1];b={s:n.body(f'{s}_sharpa_wave_{s}_hand_C_MC').id for s in('left','right')};F=min(len(J),len(P));pal={s:np.zeros((F,3)) for s in b}
        for f in range(F):
            d.qpos[:nq]=J[f];mujoco.mj_kinematics(n,d)
            for s in b:pal[s][f]=d.xpos[b[s]]
        out=[]
        for k in range(P.shape[1]):
            o=P[:F,k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();g=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);mv=np.linalg.norm(g-g[0],axis=1)>.03;c={s:float(np.median(np.linalg.norm(pal[s][mv]-g[mv],axis=1))) if mv.any() else 9. for s in b};side=min(c,key=c.get);frac={s:round(float((np.linalg.norm(pal[s][mv]-g[mv],axis=1)<.2).mean()),2) if mv.any() else 0. for s in b};out.append(dict(obj=k,side=side,right_cm=round(100*c['right']),left_cm=round(100*c['left']),near_frac=frac,moved_s=round(float(mv.sum())/20,1),first_s=round(float(np.argmax(mv))/20,2) if mv.any() else -1))
        print(json.dumps(dict(ep=e.name[-3:],frames=F,objects=out,yam=(e/'yam/yam_model.mjb').exists())))
    except Exception as x:print(json.dumps(dict(ep=e.name[-3:],error=repr(x)[:100])))
