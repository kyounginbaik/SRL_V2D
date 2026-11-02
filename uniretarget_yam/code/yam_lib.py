"""Can a YAM arm follow the floating-hand tapes? IK over every frame, with limits and speed."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from fastsim import FastSim
Y='/data5/exaFLOPs/proj/uniretarget/furniturebench_yam/src/furniturebench_yam/assets/data/urdf/yam_sharpa_%s.urdf'
class Arm:
    def __init__(self,side):
        self.m=mujoco.MjModel.from_xml_path(Y%side);self.d=mujoco.MjData(self.m);self.j=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}') for i in range(1,7)];self.q=[int(self.m.jnt_qposadr[j]) for j in self.j];self.v=[int(self.m.jnt_dofadr[j]) for j in self.j];self.b=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'link_6');self.tool_p=np.array([0.,0.,.147+.0005+.029]);self.tool_r=R.from_euler('z',np.pi if side=='left' else 0.);self.lo=self.m.jnt_range[self.j,0];self.hi=self.m.jnt_range[self.j,1]
    def fk(self,q):
        self.d.qpos[self.q]=q;mujoco.mj_kinematics(self.m,self.d);rl=R.from_matrix(self.d.xmat[self.b].reshape(3,3).copy());return self.d.xpos[self.b]+rl.apply(self.tool_p),rl*self.tool_r
    def ik(self,q,p,r,it=40):
        q=q.copy();jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        for _ in range(it):
            pc,rc=self.fk(q);e=np.r_[p-pc,.3*(r*rc.inv()).as_rotvec()]
            if np.linalg.norm(e[:3])<5e-4 and np.linalg.norm(e[3:])<.002:break
            mujoco.mj_comPos(self.m,self.d);mujoco.mj_jac(self.m,self.d,jp,jr,pc,self.b);J=np.r_[jp[:,self.v],.3*jr[:,self.v]];q=np.clip(q+.7*J.T@np.linalg.solve(J@J.T+1e-3*np.eye(6),e),self.lo,self.hi)
        pc,rc=self.fk(q);return q,float(np.linalg.norm(p-pc)),float((r*rc.inv()).magnitude())
def wrist_world(ep_dir,plan,side):
    e=Path(ep_dir);s=FastSim(e/'native_model.mjb',e/'baseline/physics.npz');m=s.m;d=s.d;b=[i for i in range(m.nbody) if m.body(i).name==f'{side}_sharpa_wave_{side}_hand_C_MC'][0];Q=np.load(plan)['joint_q'][0];P=[];Rs=[]
    for f in range(len(Q)):
        d.qpos[:]=s.native(Q[f]);mujoco.mj_kinematics(m,d);P.append(d.xpos[b].copy());Rs.append(R.from_matrix(d.xmat[b].reshape(3,3).copy()))
    up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);return np.array(P),Rs,up,s.poses,float(s.fps)
