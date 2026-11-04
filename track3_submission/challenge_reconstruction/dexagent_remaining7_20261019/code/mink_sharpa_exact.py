"""Floating Sharpa adapter of the tuned Mink 1.3 setup; Pink retained for FK only."""
import runpy, json, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mink, mujoco
from scipy.spatial.transform import Rotation as R
from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
import robotic_grounding.retarget.retarget_utils as utils

class MinkSharpa(SharpaHandKinematics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tree=ET.parse(self.robot_asset_path);root=tree.getroot()
        # Kinematics only: retain inertias and sites, remove render/collision meshes.
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ('geom','asset','actuator','sensor','contact','equality','keyframe'):
                    parent.remove(child)
        body=root.find('worldbody/body');ET.SubElement(body,'freejoint',name='floating_root')
        self.mm=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.mc=mink.Configuration(self.mm)
        self.mapping=[]
        for jid in range(2,self.robot.model.njoints):
            name=self.robot.model.names[jid];mj=mujoco.mj_name2id(self.mm,mujoco.mjtObj.mjOBJ_JOINT,name)
            assert mj>=0,name
            self.mapping.append((self.robot.model.joints[jid].idx_q,int(self.mm.jnt_qposadr[mj])))
        assert self.mm.nq==self.robot.model.nq
        self.mt={}
        for name,(_,pc,oc) in self.target_to_source.items():
            scale=2. if name.endswith('C_MC') or '_tip_' in name else 1.5
            typ='site' if mujoco.mj_name2id(self.mm,mujoco.mjtObj.mjOBJ_SITE,name)>=0 else 'body'
            self.mt[name]=mink.FrameTask(name,typ,position_cost=pc*scale,orientation_cost=oc,lm_damping=1.)
        self.limits=[mink.ConfigurationLimit(self.mm)]
        self.audit=[];self.targets=[]
        # Verify frame conventions and joint ordering before any solve.
        q=self.robot.q0.copy();q[:3]=[.1,.2,.3];q[3:7]=R.from_euler('xyz',[.1,.2,.3]).as_quat()
        self.configuration.update(q);self.mc.update(self.to_mj(q))
        for name,t in self.mt.items():
            a=self.configuration.get_transform_frame_to_world(name)
            b=self.mc.get_transform_frame_to_world(name,t.frame_type)
            assert np.allclose(a.homogeneous,b.as_matrix(),atol=1e-6),name
        print('Mink FK mapping verified',self.side,flush=True)
    def to_mj(self,q):
        out=np.zeros(self.mm.nq);out[:3]=q[:3];out[3:7]=q[[6,3,4,5]]
        for p,m in self.mapping:out[m]=q[p]
        return out
    def to_pin(self,q):
        out=np.zeros(self.robot.model.nq);out[:3]=q[:3];out[3:7]=q[[4,5,6,3]]
        for p,m in self.mapping:out[p]=q[m]
        return out
    def compute(self,source_joints,source_joints_wxyz,source_to_robot_scale=1.,qpos=None):
        def arr(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
        self.set_frame_tasks_target(arr(source_joints),arr(source_joints_wxyz),source_to_robot_scale=source_to_robot_scale)
        for name,t in self.mt.items():
            target=self.frame_tasks[name].transform_target_to_world
            t.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target.rotation),target.translation.copy()))
        self.targets.append({n: (t.transform_target_to_world.rotation.copy(), t.transform_target_to_world.translation.copy()) for n,t in self.frame_tasks.items()})
        def solve(seed):
            self.mc.update(self.to_mj(seed));best=None;score=np.inf;previous=np.inf
            for it in range(self.max_iter):
                v=mink.solve_ik(self.mc,list(self.mt.values()),self.dt,solver='daqp',limits=self.limits,safety_break=False)
                assert np.isfinite(v).all()
                self.mc.integrate_inplace(v,self.dt)
                errors={n:np.asarray(t.compute_error(self.mc)) for n,t in self.mt.items()}
                value=sum(float(np.sum((e*np.asarray(self.mt[n].cost))**2)) for n,e in errors.items())
                if value<score:best=(self.mc.q.copy(),errors,it+1);score=value
                if abs(previous-value)<1e-10:break
                previous=value
            return score,best
        seed=self.robot.q0.copy() if qpos is None else qpos.copy()
        score,best=solve(seed);errors=best[1]
        retry=any(np.linalg.norm(e[:3])>(.05 if n.endswith('C_MC') else .08) for n,e in errors.items())
        if retry:
            neutral=self.robot.q0.copy();w=self.frame_tasks[self.side+'_hand_C_MC'].transform_target_to_world
            neutral[:3]=w.translation;neutral[3:7]=R.from_matrix(w.rotation).as_quat()
            s,b=solve(neutral)
            if s<score:score,best=s,b
        q=self.to_pin(best[0]);self.configuration.update(q)
        poses=[]
        for name in self.robot_frame_names.values():
            t=self.configuration.get_transform_frame_to_world(name)
            poses.append(np.r_[t.translation,R.from_matrix(t.rotation).as_quat(scalar_first=True)])
        errs=[float(np.linalg.norm(best[1][n][:3])) for n in self.mt]
        self.audit.append({'errors_m':errs,'recovery_attempted':retry})
        return dict(q=q,frame_pose=np.asarray(poses),frame_task_errors=errs,num_optimization_iterations=best[2])

