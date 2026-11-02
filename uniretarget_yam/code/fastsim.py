"""Plain-MuJoCo replay of a native target tape from the real reset (fast proposal filter)."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R

class FastSim:
    def __init__(self,model_path,physics_path,fps=20.,substeps=20,wrist_kp=100.,wrist_tau=90.,object_index=0):
        self.m=m=mujoco.MjModel.from_binary_path(str(model_path));m.opt.timestep=1/(fps*substeps);self.d=mujoco.MjData(m);self.fps=fps;self.substeps=substeps;self.kp=wrist_kp;self.tau=wrist_tau
        with np.load(physics_path) as z:
            self.jmap=z['native_joint_to_newton_joint'][0];self.qs=z['newton_joint_q_start'];self.ds=z['newton_joint_qd_start'];self.q0=z['native_qpos'][0,0].copy();self.poses=z['object_pose'][:,0].copy();self.pose=self.poses[:,object_index]
        self.robot=[j for j in range(m.njnt) if m.jnt_type[j]!=0];self.balls=[j for j in self.robot if m.jnt_type[j]==1]
        self.pos={}
        for a in range(m.nu):
            j=int(m.actuator_trnid[a,0])
            if m.actuator_gainprm[a,0]>0 and abs(m.actuator_biasprm[a,1]+m.actuator_gainprm[a,0])<1e-5:self.pos[j]=a
        # Match each task object to its native free joint by where it starts, not by order.
        free=[j for j in range(m.njnt) if m.jnt_type[j]==0];self.object_qpos=[]
        for k in range(self.poses.shape[1]):
            self.object_qpos.append(min((int(m.jnt_qposadr[j]) for j in free),key=lambda q:float(np.linalg.norm(self.q0[q:q+3]-self.poses[0,k,:3]))))
        if len(set(self.object_qpos))!=len(self.object_qpos):raise ValueError('task objects do not map one-to-one onto free joints')
        self.qo=self.object_qpos[object_index];self.obj=int(m.jnt_bodyid[[j for j in free if int(m.jnt_qposadr[j])==self.qo][0]]);self.object_index=object_index
    def native(self,q):
        out=self.m.qpos0.copy()
        for j in self.robot:
            n=int(self.jmap[j]);w=4 if self.m.jnt_type[j]==1 else 1;v=q[self.qs[n]:self.qs[n]+w];out[self.m.jnt_qposadr[j]:self.m.jnt_qposadr[j]+w]=v[[3,0,1,2]] if w==4 else v
        return out
    def reset(self,q_tape0,object_pose_xyz_wxyz=None):
        m,d=self.m,self.d;mujoco.mj_resetData(m,d);d.qpos[:]=self.native(q_tape0)
        for k,qi in enumerate(self.object_qpos):
            init=self.poses[0,k];d.qpos[qi:qi+3]=init[:3];d.qpos[qi+3:qi+7]=init[3:7][[3,0,1,2]]
        if object_pose_xyz_wxyz is not None:d.qpos[self.qo:self.qo+7]=object_pose_xyz_wxyz
        d.qvel[:]=0;mujoco.mj_forward(m,d)
    def step(self,target):
        m,d=self.m,self.d
        for j,a in self.pos.items():d.ctrl[a]=target[self.ds[self.jmap[j]]]
        goals=[(j,R.from_rotvec(target[self.ds[self.jmap[j]]:self.ds[self.jmap[j]]+3])) for j in self.balls]
        for _ in range(self.substeps):
            d.qfrc_applied[:]=0
            for j,g in goals:
                qi=int(m.jnt_qposadr[j]);di=int(m.jnt_dofadr[j])
                d.qfrc_applied[di:di+3]=np.clip(self.kp*(R.from_quat(d.qpos[qi:qi+4][[1,2,3,0]]).inv()*g).as_rotvec(),-self.tau,self.tau)
            mujoco.mj_step(m,d)
    def replay(self,q,target,frames,on_frame=None):
        self.reset(q[0]);out=np.zeros((frames,self.m.nq))
        for f in range(frames):
            self.step(target[f]);out[f]=self.d.qpos
            if on_frame is not None:on_frame(f,self.d)
        return out
