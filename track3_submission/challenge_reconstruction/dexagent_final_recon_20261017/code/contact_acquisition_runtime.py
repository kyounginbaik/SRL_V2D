"""Robot-only contact acquisition feedback with replayable 500 Hz controls.

The integration loop is a deliberately isolated copy of floating_runtime's
verified servo loop. Its sole integration change is the bounded robot-command
callback immediately before actuator control. No scene/object state, clock,
force cap, applied force, weld, or integrator state is altered by feedback.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'third_party/DexAgent'))
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from floating_runtime import SCHEMA,WRIST,_assert_contract,_freeze,_write,_sha
from wrist_reference import continuous_euler_xyz


def smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


class ContactAcquisition:
    """Acquire an object physically, then hand control back to the trajectory.

    Contact detection uses the previous completed MuJoCo step (2 ms latency).
    Object pose is read from the current unactuated free joint. Robot commands
    only are changed, and the final clip still uses the declared joint limits.
    """
    def __init__(self,model,config):
        from tool_library.common import hand as H
        self.m=model;self.c=config;self.side=config['side'];self.name=config['object_name']
        self.rig=H.HandRig.build(model,f'robot_{self.side}_',self.side)
        self.wcols=np.array([self.rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
        self.fkeys=list(self.rig.finger_qpos);self.fcols=np.array(list(self.rig.finger_qpos.values()))
        self.robot_cols=np.r_[self.wcols,self.fcols]
        self.bid=model.body(self.name).id;self.oc=int(model.joint(self.name+'_free').qposadr[0])
        self.localp=np.asarray(config['object_from_wrist_position'],float)
        self.localR=np.asarray(config['object_from_wrist_rotation'],float)
        self.closed=np.array([config['closed_finger_targets'][k] for k in self.fkeys])
        self.opened=self.closed.copy()
        for i,k in enumerate(self.fkeys):
            if H.finger_of_joint(k)=='thumb':self.opened[i]=config['open_thumb_targets'].get(k,self.closed[i])
        self.phase='waiting';self.events=[];self.frames=[];self.force=np.zeros(6)
        self.nonthumb_since=None;self.both_since=None;self.clamp_start=None;self.capture_time=None
        self.capture_pose=None;self.capture_fingers=None;self.last_euler=None;self.last_command=None;self.transport_start=None
        self.last_time=None
        self.body_finger={}
        for b in self.rig.body_ids:
            bodyname=model.body(b).name
            for f in H.FINGERS:
                if f'_{f}_' in bodyname:self.body_finger[b]=f

    def contacts(self,data):
        forces={}
        for i,c in enumerate(data.contact):
            b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])
            hand=b2 if b1==self.bid else b1 if b2==self.bid else -1
            finger=self.body_finger.get(hand)
            if finger is None:continue
            mujoco.mj_contactForce(self.m,data,i,self.force)
            forces[finger]=forces.get(finger,0.)+max(0.,float(self.force[0]))
        return forces

    def event(self,time,phase,**extra):
        self.phase=phase;self.events.append({'time_seconds':float(time),'phase':phase,**extra})

    def command(self,data,base,time):
        c=self.c;out=base.copy();start=c.get('acquisition_start_s',1.5);deadline=c.get('deadline_s',3.0)
        if time<start:return out
        if self.phase=='waiting':self.event(time,'approach');self.last_euler=base[self.wcols[3:]].copy()
        observed=self.contacts(data);threshold=c.get('min_contact_force_n',.05)
        thumb=observed.get('thumb',0.)>=threshold
        opposing_count=sum(f!='thumb' and f in c.get('opposing_fingers',['index','middle','ring']) and v>=threshold for f,v in observed.items())
        opposing=opposing_count>0
        if opposing:self.nonthumb_since=time if self.nonthumb_since is None else self.nonthumb_since
        else:self.nonthumb_since=None
        if self.phase=='approach' and self.nonthumb_since is not None and time-self.nonthumb_since>=c.get('contact_confirm_s',.02):
            self.clamp_start=time;self.event(time,'close_thumb',observed_forces_n=observed.copy())
        if self.phase=='close_thumb':
            if thumb and opposing_count>=c.get('min_opposing_fingers',1):self.both_since=time if self.both_since is None else self.both_since
            else:self.both_since=None
            if self.both_since is not None and time-self.both_since>=c.get('capture_confirm_s',.06):
                self.capture_time=time;self.capture_pose=(self.last_command[self.wcols] if self.last_command is not None else data.qpos[self.wcols]).copy();self.event(time,'dwell',observed_forces_n=observed.copy())
        if self.phase=='dwell' and time-self.capture_time>=c.get('dwell_s',.12):
            self.transport_start=time;self.event(time,'transport')
        if self.phase in ['approach','close_thumb'] and time>=deadline:
            self.capture_pose=data.qpos[self.wcols].copy();self.capture_fingers=data.qpos[self.fcols].copy();self.transport_start=time;self.event(time,'failed_acquisition')
        if self.phase in ['approach','close_thumb']:
            actual=data.qpos[self.oc:self.oc+7];R=Rotation.from_quat(actual[3:],scalar_first=True).as_matrix()
            pos=actual[:3]+R@self.localp;rot=Rotation.from_matrix(R@self.localR)
            euler=continuous_euler_xyz(Rotation.from_matrix([rot.as_matrix()]),self.last_euler)[0];self.last_euler=euler
            goal=np.r_[pos,euler]
            # Bound object pursuit to avoid chasing an ejected object forever.
            change=goal[:3]-base[self.wcols[:3]];limit=c.get('max_tracking_offset_m',.10)
            if np.linalg.norm(change)>limit:goal[:3]=base[self.wcols[:3]]+change*limit/np.linalg.norm(change)
            alpha=smooth((time-start)/c.get('approach_s',.30))
            out[self.wcols]=(1-alpha)*base[self.wcols]+alpha*goal
            close=smooth((time-self.clamp_start)/c.get('thumb_closure_s',.15)) if self.clamp_start is not None else 0.
            finger=(1-close)*self.opened+close*self.closed
            out[self.fcols]=(1-alpha)*base[self.fcols]+alpha*finger
        elif self.phase=='dwell':
            out[self.wcols]=self.capture_pose;out[self.fcols]=self.closed
        elif self.phase in ['transport','failed_acquisition']:
            alpha=smooth((time-self.transport_start)/c.get('transport_blend_s',.7))
            out[self.wcols]=(1-alpha)*self.capture_pose+alpha*base[self.wcols]
            if self.phase=='transport':out[self.fcols]=self.closed*(1-alpha)+base[self.fcols]*alpha
            else:out[self.fcols]=(1-alpha)*self.capture_fingers+alpha*base[self.fcols]
        if self.phase in ['approach','close_thumb'] and self.last_command is not None:
            dt=max(1e-6,time-self.last_time)
            previous=self.last_command[self.wcols];goal=out[self.wcols].copy()
            step=goal[:3]-previous[:3];limit=c.get('max_acquisition_linear_speed_m_s',.5)*dt
            if np.linalg.norm(step)>limit:goal[:3]=previous[:3]+step*limit/np.linalg.norm(step)
            rp=Rotation.from_euler('XYZ',previous[3:]);rn=Rotation.from_euler('XYZ',goal[3:]);delta=(rp.inv()*rn).as_rotvec()
            limit=c.get('max_acquisition_angular_speed_rad_s',3.)*dt
            if np.linalg.norm(delta)>limit:
                rn=rp*Rotation.from_rotvec(delta*limit/np.linalg.norm(delta))
                goal[3:]=continuous_euler_xyz(Rotation.from_matrix([rn.as_matrix()]),previous[3:])[0]
            out[self.wcols]=goal
        # Only robot coordinates may differ; all object reference columns remain.
        unchanged=np.setdiff1d(np.arange(self.m.nq),self.robot_cols)
        if not np.array_equal(out[unchanged],base[unchanged]):raise AssertionError('feedback changed nonrobot coordinates')
        self.last_command=out.copy()
        self.last_time=time
        self.frames.append([time,self.phase,observed.copy(),bool(thumb),bool(opposing)])
        return out

def servo_rollout_feedback(model, Qref, fps, qpos_start=None, qvel_start=None,
                  state_start=None, record_substeps=True, controller=None, **options):
    """Run position servos without state injection or changing scene physics.

    Frame zero is the initial physical state. Remaining reference rows execute
    at their original timestamps (no time stretching). Complete integration
    state can be supplied for exact MPC trial/resume. Qref object columns never
    enter ctrl. record_substeps=False only reduces diagnostic storage.
    """
    from scipy.spatial.transform import Rotation
    from tool_library.common import episode as E
    _assert_contract(model)
    for name in ("eq_inactive", "rules", "latched", "mechanism_state_start"):
        if options.get(name):
            raise ValueError(f"{name}: mechanisms are not supported by this rigid-object adapter")
    if any(options.get(k) is not None for k in ("finger_force_nm", "wrist_force_n", "wrist_torque_nm", "finger_kp")):
        raise ValueError("runtime gains/force limits are frozen in scene.xml; per-trial changes are forbidden")
    if options.get("settle_s", 0) or options.get("ctrl_lead_s", 0) or options.get("balance_grip", False):
        raise ValueError("settling, time lead and adaptive gain changes are excluded")
    Qref = np.asarray(Qref, dtype=float)
    if Qref.ndim != 2 or Qref.shape[1] != model.nq or not len(Qref) or not np.isfinite(Qref).all():
        raise ValueError("reference must be finite [T, model.nq]")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0 or fps * model.opt.timestep > 1:
        raise ValueError("fps must be positive and no faster than physics")
    data, target, measured = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(model, state_spec)
    if state_start is not None:
        raw = np.asarray(state_start, float)
        if raw.shape != (state_size,) or not np.isfinite(raw).all():
            raise ValueError("invalid complete initial integration state")
        mujoco.mj_setState(model, data, raw, state_spec)
        if qpos_start is not None and not np.array_equal(data.qpos, qpos_start):
            raise ValueError("qpos_start contradicts integration state")
        if qvel_start is not None and not np.array_equal(data.qvel, qvel_start):
            raise ValueError("qvel_start contradicts integration state")
    else:
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        if qpos_start is not None:
            data.qpos[:] = np.asarray(qpos_start, float)
        if qvel_start is not None:
            data.qvel[:] = np.asarray(qvel_start, float)
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("external force injection is forbidden")
    mujoco.mj_forward(model, data)
    initial_qpos, initial_qvel, initial_act = data.qpos.copy(), data.qvel.copy(), data.act.copy()
    initial_state = np.zeros(state_size)
    mujoco.mj_getState(model, data, initial_state, state_spec)
    initial_time = float(data.time)
    parts = E.robot_parts(model)
    ids = np.arange(model.nu)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]].astype(int)
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange.T
    T, dt = len(Qref), float(model.opt.timestep)
    Q = np.empty((T, model.nq)); V = np.empty((T, model.nv)); C = np.empty((T, model.nu))
    times = np.empty(T); tracking = np.zeros((T, 2)); forces = np.zeros((T, 2))
    pen = np.zeros(T); objpen = np.zeros(T); wrench = np.zeros((T, 2))
    contacts = []; sq = []; sv = []; sc = []; st = []
    force6 = np.zeros(6); tangent = np.zeros(model.nv)
    robot = {b for b in range(model.nbody) if (model.body(b).name or "").startswith("robot_")}
    side_of_body = {b: si for si, s in enumerate(("left", "right")) for b in parts[s]["rig"].body_ids}
    slide_aids = [a for a in ids if model.jnt_type[int(model.actuator_trnid[a, 0])] == mujoco.mjtJoint.mjJNT_SLIDE]
    rot_aids = [a for a in ids if any((model.joint(int(model.actuator_trnid[a, 0])).name or "").endswith(s) for s in WRIST[3:])]

    def diagnose(t):
        touched, net = set(), np.zeros((2, 3))
        for ci in range(data.ncon):
            contact = data.contact[ci]
            b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            r1, r2 = b1 in robot, b2 in robot
            depth = max(0., -float(contact.dist))
            if r1 != r2:
                pen[t] = max(pen[t], depth)
                hand, obj = (b1, b2) if r1 else (b2, b1)
                si = side_of_body.get(hand)
                if si is not None:
                    touched.add((("left", "right")[si], model.body(obj).name))
                    mujoco.mj_contactForce(model, data, ci, force6)
                    net[si] += contact.frame.reshape(3, 3).T @ force6[:3] * (1 if r1 else -1)
            elif not r1 and not r2:
                objpen[t] = max(objpen[t], depth)
        forces[t] = np.maximum(forces[t], np.linalg.norm(net, axis=1))
        wrench[t] = np.maximum(wrench[t], [np.max(np.abs(data.actuator_force[slide_aids]), initial=0.),
                                           np.max(np.abs(data.actuator_force[rot_aids]), initial=0.)])
        return sorted(touched)

    unstable = None
    completed = 0
    for t in range(T):
        previous, current = round(max(0, t - 1) / (fps * dt)), round(t / (fps * dt))
        n = current - previous
        for k in range(n):
            # Interpolate on generalized coordinates; only robot joints command actuators.
            mujoco.mj_differentiatePos(model, tangent, 1., Qref[t - 1], Qref[t])
            target.qpos[:] = Qref[t - 1]
            mujoco.mj_integratePos(model, target.qpos, tangent, (k + 1) / n)
            if controller is not None:
                target.qpos[:] = controller.command(data, target.qpos, float(data.time) - initial_time)
            data.ctrl[:] = np.clip(target.qpos[qadr], ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)
            if record_substeps:
                sq.append(data.qpos.copy()); sv.append(data.qvel.copy()); sc.append(data.ctrl.copy())
                st.append(float(data.time) - initial_time)
            # Do not call mj_forward here: the exact integrator warmstart state is preserved.
            if record_substeps:
                diagnose(t)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
                unstable = {"frame": t, "reason": "nonfinite state or BADQACC reset"}
                break
        Q[t], V[t], C[t], times[t] = data.qpos, data.qvel, data.ctrl, float(data.time) - initial_time
        # Kinematics only on copied data: diagnostics cannot perturb simulation warmstart.
        measured.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, measured)
        target.qpos[:] = controller.last_command if controller is not None and controller.last_command is not None else Qref[t]
        mujoco.mj_kinematics(model, target)
        for side, part in parts.items():
            body = part["ee_body"]
            p = np.linalg.norm(target.xpos[body] - measured.xpos[body])
            R = target.xmat[body].reshape(3, 3).T @ measured.xmat[body].reshape(3, 3)
            tracking[t] = np.maximum(tracking[t], [p, np.degrees(Rotation.from_matrix(R).magnitude())])
        contacts.append(diagnose(t)); completed = t + 1
        if unstable:
            break
    final_state = np.zeros(state_size); mujoco.mj_getState(model, data, final_state, state_spec)
    sl = slice(0, completed)
    return {"qpos": Q[sl], "qvel": V[sl], "ctrl": C[sl], "time_s": times[sl],
            "initial_qpos": initial_qpos, "initial_qvel": initial_qvel, "initial_act": initial_act,
            "initial_state": initial_state, "final_state": final_state, "state_spec": int(state_spec),
            "initial_time_s": initial_time, "final_qpos": data.qpos.copy(), "final_qvel": data.qvel.copy(),
            "track_pos_m": tracking[sl, 0], "track_rot_deg": tracking[sl, 1],
            "hand_pen_m": pen[sl], "object_pen_m": objpen[sl], "hand_object_force_n": forces[sl],
            "wrist_force_n": wrench[sl, 0], "wrist_torque_nm": wrench[sl, 1],
            "contacts": contacts, "unstable": unstable, "frames_simulated": completed,
            "robot_schema": SCHEMA, "eq_inactive": [], "released": [], "latched": {}, "latch_events": [],
            "mechanism_initial": {}, "mechanism_final": {}, "gains": {"source": "frozen_scene"},
            "substep_qpos": np.asarray(sq).reshape(-1, model.nq),
            "substep_qvel": np.asarray(sv).reshape(-1, model.nv),
            "substep_ctrl": np.asarray(sc).reshape(-1, model.nu), "substep_time_s": np.asarray(st)}


def execute_acquisition(scene,reference,out_dir,config):
    """Execute robot-only feedback and independently replay actual controls."""
    from tool_library.common import tracking as TR
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'trajectory.npz').exists():raise FileExistsError('select a fresh output directory')
    model,frozen=_freeze(scene,out)
    ref=TR.load_reference(reference,model);Qref=ref['qpos'];fps=ref['fps']
    start=mujoco.MjData(model)
    if model.nkey:mujoco.mj_resetDataKeyframe(model,start,0)
    # The reference is an actuator command, not an injected state. MuJoCo's
    # XML serialization rounds some keyframe values; physical initialization
    # always uses the exact frozen scene keyframe, as execute_reference does.
    initial_command_delta=float(np.max(np.abs(Qref[0]-start.qpos)))
    controller=ContactAcquisition(model,config)
    diag=servo_rollout_feedback(model,Qref,fps,qpos_start=start.qpos.copy(),controller=controller)
    if not np.array_equal(diag['initial_qpos'],start.qpos):raise AssertionError('physical initial state differs from scene keyframe')
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].astype(int)
    executed=Qref[:len(diag['qpos'])].copy();executed[1:,qadr]=diag['ctrl'][1:]
    diag['executed_command_qpos']=executed
    report=_write(model,frozen,out,diag,Qref,ref['labels'],fps,
                  {'reference_source':str(reference),'feedback_adapter':str(Path(__file__).resolve()),
                   'feedback_adapter_sha256':_sha(__file__),'feedback_config':config,
                   'feedback_events':controller.events,'physical_capture_detected':controller.capture_time is not None,
                   'feedback_observation':'current object free-joint pose; previous completed physical step finger/object normal forces',
                   'feedback_rate_hz':1./model.opt.timestep,'recorded_control_rate_hz':1./model.opt.timestep,
                   'physical_initial_state_exact_scene_keyframe':True,'initial_reference_keyframe_max_coordinate_difference':initial_command_delta,
                   'original_clock_preserved':True,'object_actuation':False,'no_retiming':True})
    TR.save_reference(out/'executed_command_reference.npz',model,executed,ref['labels'],fps,ref['segment'],ref['segments'],
                      source='actual bounded robot servo commands from contact acquisition; original object targets retained',arms_coupled=False)
    (out/'acquisition_events.json').write_text(json.dumps({'events':controller.events,'config':config,'samples':controller.frames},indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(execute_acquisition(a.scene,a.reference,a.out,json.loads(a.config.read_text())),indent=2))
