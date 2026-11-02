"""Inverse dynamics expressed through ordinary native scalar PD targets.

All state lives in private planning data; no object force or simulator state is
injected into execution. Ball/orientation drives require explicit native controller gains and limits.
"""
from dataclasses import dataclass,field
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ScalarPDInverseResult:
    position_controls: dict
    native_joint_position_targets: dict
    required_generalized_effort: np.ndarray
    unhandled_native_dofs: np.ndarray
    violations: tuple
    native_ball_rotvec_targets: dict = field(default_factory=dict)


class NativeScalarPDInverse:
    def __init__(self,model,ball_drives=None):
        import mujoco
        self.model=model;self.data=mujoco.MjData(model);self.position={}
        self.ball_drives={} if ball_drives is None else dict(ball_drives)
        for joint,(kp,limit) in self.ball_drives.items():
            if joint not in range(model.njnt) or model.jnt_type[joint]!=1 or not np.isfinite([kp,limit]).all() or min(kp,limit)<=0:
                raise ValueError('explicit positive native ball-controller gains and effort limits required')
        for a in range(model.nu):
            joint=int(model.actuator_trnid[a,0]);gear=model.actuator_gear[a]
            if (model.actuator_trntype[a]!=mujoco.mjtTrn.mjTRN_JOINT or joint<0
                    or model.jnt_type[joint] not in (2,3) or abs(gear[0]-1)>1e-8
                    or np.linalg.norm(gear[1:])>1e-8):continue
            kp=float(model.actuator_gainprm[a,0])
            if kp>0 and abs(model.actuator_biasprm[a,1]+kp)<1e-5:
                if joint in self.position:raise ValueError('multiple position drives per joint need an explicit effort allocation')
                self.position[joint]=(a,kp)

    def targets(self,qpos,qvel=None,qacc=None,contact_load=None):
        """Compute q_PD = q_geometry + required_motor_load / native gain.

        contact_load is the robot generalized load required to oppose the
        object reaction, e.g. sum J_hand.T @ force_ON_OBJECT. Passive forces,
        gravity, inertia and native velocity-servo damping are accounted for.
        Other scalar actuator control inputs are assumed zero, as in the
        verified native zero-residual controller. Violations are reported;
        target positions are never clipped to mechanical coordinate stops.
        """
        import mujoco
        m,d=self.model,self.data
        q=np.asarray(qpos,float);v=np.zeros(m.nv) if qvel is None else np.asarray(qvel,float)
        a=np.zeros(m.nv) if qacc is None else np.asarray(qacc,float)
        load=np.zeros(m.nv) if contact_load is None else np.asarray(contact_load,float)
        if q.shape!=(m.nq,) or any(x.shape!=(m.nv,) for x in (v,a,load)) or not all(np.isfinite(x).all() for x in (q,v,a,load)):
            raise ValueError('finite native position/velocity/acceleration/load vectors required')
        d.qpos[:]=q;d.qvel[:]=v
        mujoco.mj_fwdPosition(m,d);mujoco.mj_fwdVelocity(m,d);mujoco.mj_passive(m,d)
        d.qacc[:]=a;rne=np.zeros(m.nv);mujoco.mj_rne(m,d,1,rne)
        # jnt_actgravcomp dofs receive qfrc_gravcomp through qfrc_actuator, not
        # qfrc_passive. The motor must not supply that compensation again.
        routed=np.asarray(m.jnt_actgravcomp,bool)[m.dof_jntid]
        required=rne-d.qfrc_passive-routed*d.qfrc_gravcomp+load
        controls={};joint_targets={};handled=[];violations=[]
        for joint,(actuator,kp) in self.position.items():
            dof=int(m.jnt_dofadr[joint]);qi=int(m.jnt_qposadr[joint]);other_force=0.
            for b in range(m.nu):
                if b==actuator or m.actuator_trntype[b]!=mujoco.mjtTrn.mjTRN_JOINT or int(m.actuator_trnid[b,0])!=joint:continue
                if abs(m.actuator_gear[b,0]-1)>1e-8 or np.linalg.norm(m.actuator_gear[b,1:])>1e-8:
                    raise ValueError('unmodeled scalar actuator transmission')
                force=float(m.actuator_biasprm[b,0]+m.actuator_biasprm[b,1]*q[qi]+m.actuator_biasprm[b,2]*v[dof])
                if m.actuator_forcelimited[b]:force=float(np.clip(force,*m.actuator_forcerange[b]))
                other_force+=force
            needed=float(required[dof]-other_force)
            # At the desired state, native p-actuator force is kp*ctrl+bias.
            bias=float(m.actuator_biasprm[actuator,0]+m.actuator_biasprm[actuator,1]*q[qi]+m.actuator_biasprm[actuator,2]*v[dof])
            control=(needed-bias)/kp
            if m.actuator_forcelimited[actuator] and not m.actuator_forcerange[actuator,0]-1e-8<=needed<=m.actuator_forcerange[actuator,1]+1e-8:
                violations.append(dict(actuator=actuator,reason='native_position_effort_limit',required=needed))
            if m.actuator_ctrllimited[actuator] and not m.actuator_ctrlrange[actuator,0]-1e-8<=control<=m.actuator_ctrlrange[actuator,1]+1e-8:
                violations.append(dict(actuator=actuator,reason='native_position_control_limit',required=control))
            controls[actuator]=float(control);joint_targets[joint]=float(control);handled.append(dof)
        ball_targets={}
        for joint,(kp,limit) in self.ball_drives.items():
            qi=int(m.jnt_qposadr[joint]);di=int(m.jnt_dofadr[joint]);torque=required[di:di+3]
            if np.any(np.abs(torque)>limit+1e-8):violations.append(dict(joint=joint,reason='native_ball_effort_limit',required=torque.tolist()))
            error=torque/kp
            if np.linalg.norm(error)>=np.pi-1e-6:violations.append(dict(joint=joint,reason='native_ball_error_branch_cut'))
            # The verified runtime uses child-frame error log(cur^-1 * target).
            # Right multiplication therefore realizes the requested native
            # child-frame torque using the unchanged wrist PD controller.
            current=Rotation.from_quat(q[qi:qi+4][[1,2,3,0]])
            ball_targets[joint]=(current*Rotation.from_rotvec(error)).as_rotvec()
            handled.extend(range(di,di+3))
        return ScalarPDInverseResult(controls,joint_targets,required.copy(),
            np.setdiff1d(np.arange(m.nv),handled),tuple(violations),ball_targets)
