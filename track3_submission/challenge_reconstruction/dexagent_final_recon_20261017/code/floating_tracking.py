"""Floating-Sharpa runtime port of DexAgent's predictive-sampling MPC.

The upstream sampling_mpc_refinement algorithm is unchanged. Its six arm-joint
residuals per hand become three wrist translations and three wrist angles; its
seventh residual remains shared finger flexion. All objects are scored, including
stationary ones. No retiming, object actuation, grasp welds or pose projection.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[3]
DEX=ROOT/'third_party/DexAgent'
sys.path.insert(0,str(DEX))
from tool_library.common import tracking as TR
from tool_library.common import episode as E
from floating_runtime import servo_rollout, floating_runtime_check


class FloatingRuntimeSim(TR.RuntimeSim):
    def rollout(self,t0,t1,commands,state=None,mechanism=None):
        kw={'state_start':np.asarray(state,float)} if state is not None else {'qpos_start':self.base[0]}
        r=servo_rollout(self.model,np.asarray(commands,float),self.fps,record_substeps=False,**kw)
        return {k:r[k] for k in ['qpos','qvel','final_state','mechanism_final','unstable','hand_pen_m','hand_object_force_n']}


class FloatingBenchTracker:
    def __init__(self,scene,reference,work_dir,scene_recon_json=None,plan_arm_paths=True):
        self.work=Path(work_dir);self.work.mkdir(parents=True,exist_ok=True)
        self.scene=str(Path(scene).resolve())
        self.model=mujoco.MjModel.from_xml_path(self.scene)
        self.ref=TR.load_reference(reference,self.model)
        self.fps=self.ref['fps']
        self.Qc=self.ref['qpos'].copy()
        self.labels=list(self.ref['labels'])
        self.segment=self.ref['segment'].copy()
        self.couple_report={'ok':True,'embodiment':'floating_sharpa',
            'arms':'not present in this explicitly adapted benchmark',
            'retimed_s':0.0,'stretched':0,'original_frame_count':len(self.Qc)}
        bodies=E.object_free_joints(self.model)
        if not bodies:raise ValueError('Scene has no free object bodies')
        wrist={};flex={}
        for side in ['left','right']:
            wrist[side]=[];flex[side]=[]
            for suffix in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']:
                ids=[j for j in range(self.model.njnt)
                     if self.model.joint(j).name.endswith(f'{side}_{suffix}')]
                if len(ids)!=1:raise ValueError(f'Ambiguous wrist joint {side} {suffix}: {ids}')
                wrist[side].append(int(self.model.jnt_qposadr[ids[0]]))
            for a in range(self.model.nu):
                j=int(self.model.actuator_trnid[a,0]);name=self.model.joint(j).name
                if side in name and TR.FLEX.search(name):flex[side].append(int(self.model.jnt_qposadr[j]))
            if not flex[side]:raise ValueError(f'No flexion actuators for {side}')
        T=len(self.Qc);data=mujoco.MjData(self.model)
        bids=[self.model.body(n).id for n in bodies]
        tpos=np.zeros((T,len(bodies),3));tquat=np.zeros((T,len(bodies),4))
        for t,q in enumerate(self.Qc):
            data.qpos[:]=q;mujoco.mj_kinematics(self.model,data)
            tpos[t]=data.xpos[bids];tquat[t]=data.xquat[bids]
        self.spec={'runtime_xml':self.scene,'fps':self.fps,'base':self.Qc,'rules':[],
            'bodies':list(bodies),'joints':[],'target_pos':tpos,'target_quat':tquat,
            'target_joint':np.zeros((T,0)),'sides':['left','right'],
            'arm_cols':wrist,'flex_cols':flex}
        self.sim=FloatingRuntimeSim(self.spec)

    def export(self,path,commands,source,**extra):
        Q=np.asarray(commands,float).copy()
        for adr in E.object_free_joints(self.model).values():Q[:,adr:adr+7]=self.Qc[:,adr:adr+7]
        return TR.save_reference(path,self.model,Q,self.labels,self.fps,self.segment,
            self.ref['segments'],source,arms_coupled=False,
            floating_runtime=True,no_retiming=True,**extra)


def floating_worker_init(spec):
    TR._WORKER['sim']=FloatingRuntimeSim(spec)


def install_tracking_adapter():
    TR.BenchTracker=FloatingBenchTracker
    TR.worker_init=floating_worker_init


def refine(workspace,scene_semantic_json,scene_recon_json,reference,scene,out_dir,
           samples=16,workers=4,max_minutes=10.,seed=44,horizon_s=.4,replan_s=.2,knot_s=.2,window=None):
    """Run the released MPC implementation with an explicit floating runtime port."""
    install_tracking_adapter()
    from tool_library.skills.sampling_mpc_refinement import sampling_mpc_refinement
    result=sampling_mpc_refinement(workspace=workspace,scene_semantic_json=scene_semantic_json,
        scene_recon_json=scene_recon_json,reference=reference,scene=scene,out_dir=out_dir,
        samples=samples,workers=workers,max_minutes=max_minutes,seed=seed,
        horizon_s=horizon_s,replan_s=replan_s,knot_s=knot_s,window=window,
        sigma_arm_rad=.01,sigma_grip_rad=.05,max_arm_rad=.15,max_grip_rad=.4)
    result['adaptation']={'name':'DexAgent predictive sampling with floating Sharpa hands',
        'algorithm':'Unmodified upstream sampling_mpc_refinement',
        'upstream_commit':'e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633',
        'residual_per_hand':['wrist_x_m','wrist_y_m','wrist_z_m','wrist_rx_rad','wrist_ry_rad','wrist_rz_rad','finger_flexion_rad'],
        'robot_dofs':56,'search_dimension_per_knot':14,
        'timing':'Original full reference timeline, no retiming or approach padding',
        'scored_objects':'All free object bodies, including stationary objects',
        'search_cost':'Upstream position/rotation/contact/penetration/residual objective; MP-SR measured separately',
        'native_arm_stage3_certificate':False}
    Path(out_dir,'report.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--args',type=Path,required=True)
    a=p.parse_args();print(json.dumps(refine(**json.loads(a.args.read_text())),indent=2))
