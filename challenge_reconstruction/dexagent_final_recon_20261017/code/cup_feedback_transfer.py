"""Episode10 acquisition using an observed episode8 cup grasp.

This controls only the robot. Optional placement clearance is a wrist-command
offset; original object evaluation targets and physical initial state stay fixed.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR, hand as H
from contact_retention_runtime import ContactRetention
from contact_acquisition_runtime import servo_rollout_feedback,smooth
from floating_runtime import _freeze,_write,_sha
from evaluate_mp_sr import evaluate_files


class CupRetention(ContactRetention):
    def command(self,data,base,time):
        out=super().command(data,base,time)
        if self.phase=='transport':
            w=smooth((time-2.2)/.5)*(1-smooth((time-4.5)/.3))
            out[self.wcols[2]]+=self.c['placement_clearance_m']*w
            self.last_command=out.copy()
        return out


def run(anchor,clearance_m,preload,deadline):
    ep='episode_000010';src=R/'block_polish/episode_000008';inp=B/'episodes'/ep
    suffix='' if deadline==1.95 else f'_deadline{deadline:.2f}'
    out=R/'cup_feedback'/ep/f'a{anchor:.1f}_clear{clearance_m:.3f}_pre{preload:.2f}{suffix}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError(out)
    (out/'execution').mkdir(exist_ok=True)
    model,frozen=_freeze(R/'cem_v2'/ep/'execution/scene_physics.xml',out/'execution')
    sm=mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    rig=H.HandRig.build(sm,'robot_right_','right')
    wcols=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
    sr=TR.load_reference(src/'reference.npz',sm);z=np.load(src/'execution/trajectory.npz')
    k=round(anchor*sr['fps']);actual=z['qpos'][k];a=int(sm.joint('beige_cup_free').qposadr[0])
    orot=Rotation.from_quat(actual[a+3:a+7],scalar_first=True)
    wr=Rotation.from_euler('XYZ',actual[wcols[3:]])
    config=dict(side='right',object_name='beige_cup',
        object_from_wrist_position=orot.inv().apply(actual[wcols[:3]]-actual[a:a+3]).tolist(),
        object_from_wrist_rotation=(orot.inv()*wr).as_matrix().tolist(),
        closed_finger_targets={key:float(actual[col]+np.clip(sr['qpos'][k,col]-actual[col],-preload,preload)) for key,col in rig.finger_qpos.items()},
        open_thumb_targets={key:float(sr['qpos'][round(1.5*sr['fps']),col]) for key,col in rig.finger_qpos.items() if H.finger_of_joint(key)=='thumb'},
        opposing_fingers=['index','middle','ring','pinky'],min_opposing_fingers=1,
        acquisition_start_s=.65,deadline_s=deadline,approach_s=.35,thumb_closure_s=.15,
        capture_confirm_s=.04,contact_confirm_s=.02,dwell_s=.08,transport_blend_s=.5,
        min_contact_force_n=.1,max_tracking_offset_m=.12 if deadline==1.95 else .25,max_acquisition_linear_speed_m_s=.5,
        max_acquisition_angular_speed_rad_s=3.,require_opposed_contact_geometry=True,
        release_start_s=4.6,release_blend_s=.35,placement_clearance_m=clearance_m)
    reference=R/'continuous_references'/f'{ep}.npz';ref=TR.load_reference(reference,model);Q=ref['qpos']
    start=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,start,0)
    ctrl=CupRetention(model,config)
    diag=servo_rollout_feedback(model,Q,ref['fps'],qpos_start=start.qpos.copy(),controller=ctrl)
    assert np.array_equal(diag['initial_qpos'],start.qpos)
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].astype(int)
    executed=Q.copy();executed[1:,qadr]=diag['ctrl'][1:];diag['executed_command_qpos']=executed
    ex=_write(model,frozen,out/'execution',diag,Q,ref['labels'],ref['fps'],
        dict(reference_source=str(reference),feedback_config=config,feedback_events=ctrl.events,
             physical_capture_detected=ctrl.capture_time is not None,capture_geometry_qualified=ctrl.qualified_capture,
             feedback_adapter_sha256=_sha(__file__),object_actuation=False,original_clock_preserved=True))
    TR.save_reference(out/'executed_command_reference.npz',model,executed,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Robot-only cup acquisition feedback, unmodified source object targets',floating_runtime=True)
    manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    report=dict(episode=ep,source_anchor_s=anchor,source_recording=str(src/'execution/trajectory.npz'),
        execution=ex,metrics=metrics,config=config,learned_policy=False,original_object_targets_preserved=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'contact_geometry.json').write_text(json.dumps(ctrl.contact_geometry))
    print(json.dumps(dict(output=str(out),events=ctrl.events,mp_sr=metrics['reference_proxy_mp_sr'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--anchor',type=float,required=True)
    p.add_argument('--clearance-m',type=float,required=True);p.add_argument('--preload',type=float,default=.04)
    p.add_argument('--deadline',type=float,default=1.95)
    run(**vars(p.parse_args()))
