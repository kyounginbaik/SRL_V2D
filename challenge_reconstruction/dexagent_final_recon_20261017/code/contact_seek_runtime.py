"""Bounded robot-only contact seeking on measured collision geometry.

The additional wrist displacement is limited to 2 cm/s and 1 cm of total
path length. Object SE(3) following remains active. No object states, forces,
gains, friction, masses, timing, or effort limits are changed.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from contact_retention_runtime import ContactRetention
from contact_acquisition_runtime import servo_rollout_feedback
from floating_runtime import _freeze,_write,_sha
from evaluate_mp_sr import evaluate_files
from audit_acquisition import audit


class ContactSeeking(ContactRetention):
    def __init__(self,model,config):
        super().__init__(model,config)
        self.original_localp=self.localp.copy();self.seek_local=np.zeros(3)
        self.seek_path=0.;self.seek_stopped=False;self.seek_samples=[];self.seek_last_time=None
        self.seek_last_direction=None;self.seek_probe=mujoco.MjData(model)
        self.seek_geoms=sorted(self.rig.tip_geoms(config.get('seeking_finger','index')))
        self.object_geoms=[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==self.bid and model.geom_contype[g]]
        self.seek_force=np.zeros(6)
        if not self.seek_geoms or not self.object_geoms:raise ValueError('actual collision geometries required')
        if not (0<config.get('seeking_speed_m_s',.02)<=.02):raise ValueError('seeking speed exceeds 2 cm/s')
        if not (0<config.get('seeking_travel_limit_m',.01)<=.01):raise ValueError('seeking travel exceeds 1 cm')

    def closest_patch(self,data):
        # Distance diagnostics use a separate FK copy. H.geom_distance's tiny
        # diagnostic nudges for MuJoCo false-zero recovery never touch live data.
        probe=self.seek_probe;probe.qpos[:]=data.qpos;mujoco.mj_kinematics(self.m,probe)
        best=None;invalid=0
        for handgeom in self.seek_geoms:
            for objectgeom in self.object_geoms:
                points=np.zeros(6)
                distance=H.geom_distance(self.m,probe,handgeom,objectgeom,.05,points)
                vector=points[3:]-points[:3];length=float(np.linalg.norm(vector))
                if not np.isfinite(distance) or not np.isfinite(points).all() or length<1e-9:
                    invalid+=1;continue
                if best is None or distance<best['distance_m']:
                    best=dict(distance_m=float(distance),index_geom=self.m.geom(handgeom).name,
                        object_geom=self.m.geom(objectgeom).name,hand_point=points[:3].tolist(),
                        object_point=points[3:].tolist(),direction=(vector/length).tolist())
        return best,invalid

    def table_forces(self,data):
        total=0.;maximum=0.;contacts=[]
        for index,contact in enumerate(data.contact):
            a,b=int(self.m.geom_bodyid[contact.geom1]),int(self.m.geom_bodyid[contact.geom2])
            if a in self.rig.body_ids and self.m.body(b).name=='table':hand=a
            elif b in self.rig.body_ids and self.m.body(a).name=='table':hand=b
            else:continue
            mujoco.mj_contactForce(self.m,data,index,self.seek_force)
            force=max(0.,float(self.seek_force[0]));total+=force;maximum=max(maximum,force)
            contacts.append(dict(hand_body=self.m.body(hand).name,normal_force_n=force,
                                 penetration_m=max(0.,-float(contact.dist))))
        return dict(total_normal_force_n=total,max_normal_force_n=maximum,contacts=contacts)

    def command(self,data,base,time):
        start=self.c['acquisition_start_s']+self.c.get('approach_s',.3)
        dt=float(self.m.opt.timestep) if self.seek_last_time is None else max(0.,time-self.seek_last_time)
        self.seek_last_time=time
        observed=self.contacts(data);finger=self.c.get('seeking_finger','index')
        force=observed.get(finger,0.);patch=None;invalid=0;increment=np.zeros(3)
        if self.phase=='approach' and time>=start and time<self.c['deadline_s']:
            if not self.seek_stopped and force>=self.c.get('min_contact_force_n',.05):
                self.seek_stopped=True
                self.events.append(dict(time_seconds=float(time),phase='seeking_stopped_at_contact',
                    normal_force_n=float(force),additional_travel_m=float(self.seek_path)))
                # Stop seeking and start thumb closure atomically. Waiting for
                # another approach confirmation after stopping loses a brief
                # force-bearing contact before the thumb is allowed to move.
                self.clamp_start=time
                self.event(time,'close_thumb',observed_forces_n=observed.copy(),
                           trigger='force-bearing contact from bounded seeking')
            if not self.seek_stopped and self.seek_path<self.c.get('seeking_travel_limit_m',.01):
                patch,invalid=self.closest_patch(data)
                if patch is not None:
                    if patch['distance_m']>0:self.seek_last_direction=np.asarray(patch['direction'])
                    # Once geometrically touching, use the previous approach
                    # normal until the real normal force reaches the threshold.
                    if self.seek_last_direction is not None:
                        step=min(self.c.get('seeking_speed_m_s',.02)*dt,
                                 self.c.get('seeking_travel_limit_m',.01)-self.seek_path)
                        increment=self.seek_last_direction*step
                        R=Rotation.from_quat(data.qpos[self.oc+3:self.oc+7],scalar_first=True)
                        self.seek_local+=R.inv().apply(increment)
                        self.seek_path+=float(np.linalg.norm(increment))
                        self.localp=self.original_localp+self.seek_local
        command=super().command(data,base,time)
        if self.c['acquisition_start_s']<=time<=self.c['release_start_s']:
            self.seek_samples.append(dict(time_seconds=float(time),phase=self.phase,
                measured_index_normal_force_n=float(force),closest_patch=patch,invalid_distance_queries=invalid,
                additional_world_step_m=increment.tolist(),accumulated_additional_travel_m=float(self.seek_path),
                additional_object_local_offset_m=self.seek_local.tolist(),seeking_stopped=self.seek_stopped,
                hand_table=self.table_forces(data),
                actual_minus_commanded_wrist_translation_m=(data.qpos[self.wcols[:3]]-command[self.wcols[:3]]).tolist()))
        return command


def execute_seeking(scene,reference,out_dir,config):
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'trajectory.npz').exists():raise FileExistsError('choose a fresh output directory')
    model,frozen=_freeze(scene,out);ref=TR.load_reference(reference,model);Q=ref['qpos'];fps=ref['fps']
    start=mujoco.MjData(model)
    if model.nkey:mujoco.mj_resetDataKeyframe(model,start,0)
    controller=ContactSeeking(model,config)
    result=servo_rollout_feedback(model,Q,fps,qpos_start=start.qpos.copy(),controller=controller)
    assert np.array_equal(result['initial_qpos'],start.qpos)
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].astype(int)
    executed=Q[:len(result['qpos'])].copy();executed[1:,qadr]=result['ctrl'][1:]
    result['executed_command_qpos']=executed
    report=_write(model,frozen,out,result,Q,ref['labels'],fps,dict(
        reference_source=str(reference),feedback_adapter=str(Path(__file__).resolve()),
        feedback_adapter_sha256=_sha(__file__),feedback_config=config,feedback_events=controller.events,
        physical_capture_detected=controller.capture_time is not None,capture_geometry_qualified=controller.qualified_capture,
        physical_initial_state_exact_scene_keyframe=True,object_actuation=False,original_clock_preserved=True,
        no_retiming=True,recorded_control_rate_hz=1/model.opt.timestep,
        additional_seeking_path_m=float(controller.seek_path),seeking_contact_stop=controller.seek_stopped))
    TR.save_reference(out/'executed_command_reference.npz',model,executed,ref['labels'],fps,ref['segment'],ref['segments'],
        source='actual bounded robot control with contact seeking; original object targets unchanged',arms_coupled=False)
    (out/'acquisition_events.json').write_text(json.dumps(dict(events=controller.events,config=config,samples=controller.frames),indent=2))
    (out/'actual_contact_geometry.json').write_text(json.dumps(dict(samples=controller.contact_geometry),indent=2))
    (out/'contact_seeking_audit.json').write_text(json.dumps(dict(config=config,samples=controller.seek_samples),indent=2))
    return report


def main(tag='lid_contact_seeking_atomic'):
    source=B/'refinement_20261017/lid_collision_feedback/episode_000001/candidate_00_open10'
    config=json.loads((source/'config.json').read_text())
    config.update(seeking_finger='index',seeking_speed_m_s=.02,seeking_travel_limit_m=.01)
    out=B/f'refinement_20261017/{tag}/episode_000001/candidate_00_open10'
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(config,indent=2))
    execution=execute_seeking(source/'execution/scene_physics.xml',config['source_reference'],out/'execution',config)
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    retention=audit(out/'execution')
    report=dict(execution=execution,metrics=metrics,retention=retention)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='lid_contact_seeking_atomic')
    main(parser.parse_args().tag)
