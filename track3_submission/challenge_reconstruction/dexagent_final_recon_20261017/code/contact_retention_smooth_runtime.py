"""Isolated acquisition/retention experiment; object dynamics stay unactuated.

After observed acquisition, retain closed finger commands and the measured
wrist/object transform with capped command preload. Follow the immutable object target path with
a smooth initial SE(3) offset decay. This is a robot controller, not a reference
or physical-state edit. A contact event alone is not proof of a stable grasp.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from contact_acquisition_runtime import ContactAcquisition, servo_rollout_feedback, smooth
from floating_runtime import _freeze, _write, _sha
from wrist_reference import continuous_euler_xyz


class ContactRetention(ContactAcquisition):
    def __init__(self, model, config):
        super().__init__(model, config)
        if 'release_start_s' not in config:
            raise ValueError('explicit original release_start_s is required')
        self.retained_local = None
        self.retention_offset = None
        self.release_reported = False
        self.contact_geometry = []
        self.qualified_capture = False

    def opposing_geometry(self, data):
        """Observed, force-bearing object contacts, not planned pad targets."""
        records=[]; force=np.zeros(6)
        for i,c in enumerate(data.contact):
            b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])
            if b1==self.bid: hand,sign=b2,1.
            elif b2==self.bid: hand,sign=b1,-1.
            else: continue
            finger=self.body_finger.get(hand)
            if finger is None: continue
            mujoco.mj_contactForce(self.m,data,i,force)
            if force[0]<self.c.get('min_contact_force_n',.05): continue
            records.append((finger,np.array(c.pos),sign*np.array(c.frame[:3]),float(c.friction[0]),float(force[0])))
        best=None
        for ft,pt,nt,mt,force_t in records:
            if ft!='thumb': continue
            for ff,pf,nf,mf,force_f in records:
                if ff not in self.c.get('opposing_fingers',['index','middle','ring']): continue
                distance=np.linalg.norm(pt-pf)
                if distance<1e-5: continue
                line=(pt-pf)/distance
                nt_dot,nf_dot=float(nt@line),float(nf@(-line))
                threshold=1./np.sqrt(1.+min(mt,mf)**2)
                dot=float(nt@nf)
                margin=min(-.5-dot,nt_dot-threshold,nf_dot-threshold)
                pair={'thumb_contact_position':pt.tolist(),'opposing_contact_position':pf.tolist(),
                      'opposing_finger':ff,'normal_dot':dot,'thumb_line_cosine':nt_dot,
                      'opposing_line_cosine':nf_dot,'friction_cone_cosine':float(threshold),
                      'thumb_force_n':force_t,'opposing_force_n':force_f,'margin':float(margin),
                      'geometry_qualified':bool(margin>=0.)}
                if best is None or margin>best['margin']: best=pair
        return best

    def command(self, data, base, time):
        geometry=self.opposing_geometry(data)
        qualified=geometry is not None and geometry['geometry_qualified']
        count=self.c.get('min_opposing_fingers',1)
        required=self.c.get('require_opposed_contact_geometry',False)
        if required and not qualified: self.c['min_opposing_fingers']=999
        try:
            out = super().command(data, base, time)
        finally:
            self.c['min_opposing_fingers']=count
        if time>=self.c.get('acquisition_start_s',1.5):
            self.contact_geometry.append({'time_seconds':float(time),'pair':geometry})
        if self.capture_time is None:
            return out
        target = base[self.oc:self.oc+7]
        target_R = Rotation.from_quat(target[3:], scalar_first=True)
        if self.retained_local is None:
            self.qualified_capture=qualified
            self.original_capture_command=self.capture_pose.copy()
            actual_wrist=data.qpos[self.wcols].copy()
            delta=self.capture_pose[:3]-actual_wrist[:3]
            limit=self.c.get('capture_preload_m',.002)
            if np.linalg.norm(delta)>limit: delta*=limit/np.linalg.norm(delta)
            actual_wr=Rotation.from_euler('XYZ',actual_wrist[3:])
            delta_r=(actual_wr.inv()*Rotation.from_euler('XYZ',self.capture_pose[3:])).as_rotvec()
            limit_r=np.deg2rad(self.c.get('capture_preload_rotation_deg',3.))
            if np.linalg.norm(delta_r)>limit_r: delta_r*=limit_r/np.linalg.norm(delta_r)
            goal_wr=actual_wr*Rotation.from_rotvec(delta_r)
            self.capture_pose=np.r_[actual_wrist[:3]+delta,continuous_euler_xyz(Rotation.from_matrix([goal_wr.as_matrix()]),actual_wrist[3:])[0]]
            self.events.append({'time_seconds':float(time),'phase':'measured_capture_posture',
                                'original_command_error_m':float(np.linalg.norm(self.original_capture_command[:3]-actual_wrist[:3])),
                                'retained_preload_m':float(np.linalg.norm(delta)),
                                'retained_rotation_preload_deg':float(np.rad2deg(np.linalg.norm(delta_r)))})
            actual = data.qpos[self.oc:self.oc+7]
            actual_R = Rotation.from_quat(actual[3:], scalar_first=True)
            wrist_R = Rotation.from_euler('XYZ', self.capture_pose[3:])
            self.retained_local = (actual_R.inv().apply(self.capture_pose[:3]-actual[:3]),
                                   actual_R.inv()*wrist_R)
            self.captured_object = (actual[:3].copy(), actual_R)
            self.events.append({'time_seconds':float(time), 'phase':'retention_transform_recorded',
                                'actual_contact_geometry_qualified':qualified,'actual_contact_pair':geometry,
                                'object_from_commanded_wrist_position':self.retained_local[0].tolist(),
                                'object_from_commanded_wrist_rotation':self.retained_local[1].as_matrix().tolist()})
        if self.phase=='dwell':
            adjust=smooth((time-self.capture_time)/self.c.get('dwell_s',.12))
            old=self.original_capture_command
            out[self.wcols[:3]]=(1-adjust)*old[:3]+adjust*self.capture_pose[:3]
            oldR=Rotation.from_euler('XYZ',old[3:]); goalR=Rotation.from_euler('XYZ',self.capture_pose[3:])
            R=oldR*Rotation.from_rotvec(adjust*(oldR.inv()*goalR).as_rotvec())
            out[self.wcols[3:]]=continuous_euler_xyz(Rotation.from_matrix([R.as_matrix()]),old[3:])[0]
            self.last_command=out.copy()
        if self.phase != 'transport':
            return out
        if self.retention_offset is None:
            actual_p, actual_R = self.captured_object
            delta_R = actual_R * target_R.inv()
            self.retention_offset = (actual_p-delta_R.apply(target[:3]), delta_R.as_rotvec())
        # A fixed captured pose is the start of this C2 trajectory. Blending
        # an offset times a moving reference inherits its velocity at alpha=0
        # and would jerk the object immediately after the stationary dwell.
        u=np.clip((time-self.transport_start)/self.c.get('transport_blend_s',.7),0.,1.)
        alpha=u*u*u*(10.-15.*u+6.*u*u)
        captured_p,captured_R=self.captured_object
        desired_object_p=(1.-alpha)*captured_p+alpha*target[:3]
        desired_object_R=captured_R*Rotation.from_rotvec(alpha*(captured_R.inv()*target_R).as_rotvec())
        localp, localR = self.retained_local
        desired_wrist_R = desired_object_R*localR
        previous = self.last_euler if self.last_euler is not None else self.capture_pose[3:]
        euler = continuous_euler_xyz(Rotation.from_matrix([desired_wrist_R.as_matrix()]),previous)[0]
        self.last_euler = euler
        goal = np.r_[desired_object_p+desired_object_R.apply(localp),euler]
        release = smooth((time-self.c['release_start_s'])/self.c.get('release_blend_s',.35))
        if release>0 and not self.release_reported:
            self.events.append({'time_seconds':float(time),'phase':'release_blend'})
            self.release_reported = True
        out[self.wcols] = (1.-release)*goal+release*base[self.wcols]
        out[self.fcols] = (1.-release)*self.closed+release*base[self.fcols]
        self.last_command = out.copy()
        return out


def execute_retention(scene, reference, out_dir, config):
    from tool_library.common import tracking as TR
    out=Path(out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'trajectory.npz').exists():
        raise FileExistsError('select a fresh output directory')
    model,frozen=_freeze(scene,out)
    ref=TR.load_reference(reference,model); Qref=ref['qpos']; fps=ref['fps']
    start=mujoco.MjData(model)
    if model.nkey: mujoco.mj_resetDataKeyframe(model,start,0)
    controller=ContactRetention(model,config)
    diag=servo_rollout_feedback(model,Qref,fps,qpos_start=start.qpos.copy(),controller=controller)
    if not np.array_equal(diag['initial_qpos'],start.qpos):
        raise AssertionError('physical initial state differs from scene keyframe')
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].astype(int)
    executed=Qref[:len(diag['qpos'])].copy(); executed[1:,qadr]=diag['ctrl'][1:]
    diag['executed_command_qpos']=executed
    report=_write(model,frozen,out,diag,Qref,ref['labels'],fps,
        {'reference_source':str(reference),'feedback_adapter':str(Path(__file__).resolve()),
         'feedback_adapter_sha256':_sha(__file__),'feedback_config':config,
         'feedback_events':controller.events,'physical_capture_detected':controller.capture_time is not None,
         'capture_is_only_contact_trigger':True,'capture_geometry_qualified':controller.qualified_capture,
         'feedback_observation':'object free-joint SE(3); previous completed step finger/object normal forces',
         'feedback_rate_hz':1./model.opt.timestep,'recorded_control_rate_hz':1./model.opt.timestep,
         'physical_initial_state_exact_scene_keyframe':True,
         'initial_reference_keyframe_max_coordinate_difference':float(np.max(np.abs(Qref[0]-start.qpos))),
         'original_clock_preserved':True,'object_actuation':False,'no_retiming':True,
         'retention':'closed fingers and measured captured wrist/object transform; quintic fixed-capture to current-target pose blend; original clock and object targets'})
    TR.save_reference(out/'executed_command_reference.npz',model,executed,ref['labels'],fps,ref['segment'],ref['segments'],
        source='actual bounded robot commands from acquisition/retention; original object targets retained',arms_coupled=False)
    (out/'acquisition_events.json').write_text(json.dumps({'events':controller.events,'config':config,'samples':controller.frames},indent=2)+'\n')
    (out/'actual_contact_geometry.json').write_text(json.dumps({'criteria':'normal dot <= -0.5; contact-to-contact line inside both Coulomb friction cones; normal force threshold',
                                                             'samples':controller.contact_geometry},indent=2)+'\n')
    return report
