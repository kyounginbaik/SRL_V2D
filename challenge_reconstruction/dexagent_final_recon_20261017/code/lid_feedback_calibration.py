"""Resolve measured servo error after collision-valid grasp planning.

Only declared right-hand controller targets/gains change. Original actuator
effort limits, source masses, friction, collisions and initial states persist.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
import contact_retention_runtime as retention
from contact_acquisition_runtime import smooth
from finger_gain_sweep import finger_gain_scene
from floating_runtime import _sha
from evaluate_mp_sr import evaluate_files
from audit_acquisition import audit


class GravityRetention(retention.ContactRetention):
    def command(self,data,base,time):
        command=super().command(data,base,time)
        start=self.c['acquisition_start_s'];end=self.c['release_start_s']
        alpha=smooth((time-start)/.15)*(1-smooth((time-end)/self.c['release_blend_s']))
        if alpha<=0:return command
        aids=np.array(list(self.rig.wrist_act.values()))
        joints=self.m.actuator_trnid[aids,0];qadr=self.m.jnt_qposadr[joints];dadr=self.m.jnt_dofadr[joints]
        # Bias is read from the completed physical step: gravity/Coriolis only,
        # not applied forces or contact/constraint inverse dynamics.
        ff=np.clip(data.qfrc_bias[dadr],self.m.actuator_forcerange[aids,0],self.m.actuator_forcerange[aids,1])
        result=command.copy();result[qadr]+=alpha*ff/self.m.actuator_gainprm[aids,0]
        # Keep parent's remembered reference uncompensated; otherwise capture
        # would incorporate the motor bias and apply feedforward twice.
        return result


def run(task):
    candidate=(0,3)[task//3];variant=('gravity','finger4','gravity_finger4')[task%3]
    source=B/f'refinement_20261017/lid_collision_feedback/episode_000001/candidate_{candidate:02d}_open10'
    config=json.loads((source/'config.json').read_text())
    config.update(controller_variant=variant,controller_extension=str(Path(__file__).resolve()),
                  controller_extension_sha256=_sha(__file__),gravity_feedforward='gravity' in variant,
                  applied_external_force=False)
    out=B/f'refinement_20261017/lid_feedback_calibration/episode_000001/candidate_{candidate:02d}_{variant}'
    out.mkdir(parents=True,exist_ok=True)
    scene=source/'execution/scene_physics.xml'
    if 'finger4' in variant:
        changed=out/'controller_scene.xml'
        config['finger_controller']=finger_gain_scene(scene,changed,4.,side='right');scene=changed
    (out/'config.json').write_text(json.dumps(config,indent=2))
    # Explicit process-local dispatch extension; never changes native DexAgent
    # or another running process. The exact extension hash is in the recording.
    original=retention.ContactRetention
    try:
        if 'gravity' in variant:retention.ContactRetention=GravityRetention
        execution=retention.execute_retention(scene,config['source_reference'],out/'execution',config)
    finally:retention.ContactRetention=original
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    hold=audit(out/'execution')
    report=dict(candidate=candidate,variant=variant,config=config,execution=execution,metrics=metrics,retention=hold)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
