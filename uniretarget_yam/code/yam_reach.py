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
BASE={'left':np.array([-.31,.26,0.]),'right':np.array([-.31,-.26,0.])};READY=np.array([0.,1.3,1.3,-1.2,0.,0.])
for spec in sys.argv[1:]:
    ep,plan,side,start=spec.split(',');P,Rs,up,poses,fps=wrist_world(ep,plan,side);start=int(start);fx=np.cross([0,1,0],up);fx/=np.linalg.norm(fx);fy=np.cross(up,fx);M=np.c_[fy,-fx,up]  # robot +x = recon +y, robot +y = recon -x
    c=poses[:,:,:3].reshape(-1,3).mean(0);floor=float(min((poses[0,:,:3]@up).min(),(poses[:,:,:3].reshape(-1,3)@up).min()))-.03;arm=Arm(side);report=[]
    # Search a small set of table-centre offsets: the rule, not the episode, chooses among them.
    for dx,dy in ((0,0),(.06,0),(-.06,0),(.12,0),(0,.06),(0,-.06)):
        base_p=c-np.array([0,0,0])+M@(BASE[side]+np.array([dx,dy,0.]));base_p=base_p-(base_p@up)*up+(floor+.062)*up;BR=R.from_matrix(M);q=READY.copy();pe=[];re=[];qs=[]
        p0,r0=arm.fk(q);local=(R.from_matrix(np.eye(3)))
        for f in range(start,len(P)):
            pt=BR.inv().apply(P[f]-base_p);rt=BR.inv()*Rs[f];q,a,b=arm.ik(q,pt,rt,it=120 if f==start else 25);pe.append(a);re.append(b);qs.append(q.copy())
        pe=np.array(pe);re=np.degrees(re);qs=np.array(qs);sp=np.abs(np.diff(qs,axis=0)).max(1)*fps;ok=(pe<.01)&(re<5);report.append(dict(offset=(dx,dy),reach=round(float(ok.mean()),3),pos_p90_cm=round(float(100*np.percentile(pe,90)),1),rot_p90_deg=round(float(np.percentile(re,90)),0),speed_p90=round(float(np.percentile(sp,90)),2),speed_max=round(float(sp.max()),2),over3rad=round(float((sp>3).mean()),3)))
    report.sort(key=lambda x:-x['reach']);print(Path(ep).name[-2:],side,'frames',len(P)-start);[print('  ',x) for x in report[:3]]
